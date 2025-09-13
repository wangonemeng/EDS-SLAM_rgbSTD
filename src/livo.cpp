#include <csignal>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <deque>
#include <memory>
#include <iomanip>
#include <algorithm>

#include <Eigen/Core>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <livox_ros_driver/CustomMsg.h>

#include "so3_math.h"
#include "IMU_Processing.hpp"
#include "preprocess.h"
#include "vikit/camera_loader.h"
#include "ikd-Tree/ikd_Tree.h"
#include "visual_pnp.h"

double INIT_TIME = 0.1;
double LASER_POINT_COV = 0.001;

bool pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, align_timestamp_en = false;

float DET_RANGE = 300.0f;
float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string lid_topic, imu_topic, img_topic, config_file;

visual_pnp::VisualPNPPtr vpnp = nullptr;

int img_en = 1, lidar_en = 1, debug = 0;
double last_timestamp_lidar = 0.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
double lidar_end_time = 0.0, first_lidar_time = -1.0, first_img_time = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double voxel_grid_size = 0, filter_size_map_min = 0, cube_len = 0;
int effct_feat_num = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0;
bool point_selected_scan[100000] = {0};
bool lidar_pushed = false, flg_first_scan = true, flg_exit = false, flg_EKF_inited = false;

bool flg_first_img = true;
int grid_size, patch_size;
double cam_fx, cam_fy, cam_cx, cam_cy;

vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> cam_extrinT(3, 0.0);
vector<double> cam_extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<cv::Mat> img_buffer;
deque<double> img_time_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr scan_undistort_body(new PointCloudXYZI());
PointCloudXYZI::Ptr scan_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr scan_down_world(new PointCloudXYZI());
PointCloudXYZRGB::Ptr scan_color_world(new PointCloudXYZRGB());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> voxel_grid_scan;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
V3D Lidar_T_wrt_Cam(Zero3d);
M3D Lidar_R_wrt_Cam(Eye3d);

MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig) {
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

void pointBodyToWorld(PointType const* const pi, PointType* const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1>& pi, Matrix<T, 3, 1>& po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void pointBodyLidarToIMU(PointType const* const pi, PointType* const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void points_cache_collect() {
  PointVector points_history;
  ikdtree.acquire_removed_points(points_history);
  // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment() {
  cub_needrm.clear();
  pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
  V3D pos_LiD = pos_lid;
  if (!Localmap_Initialized) {
    for (int i = 0; i < 3; i++) {
      LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
      LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
    }
    Localmap_Initialized = true;
    return;
  }
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
  }
  if (!need_move) return;
  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points;
  float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = LocalMap_Points;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm);
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  mtx_buffer.lock();
  double preprocess_start_time = omp_get_wtime();
  if (msg->header.stamp.toSec() < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  if (p_pre->lidar_type == 3) {
    time_buffer.push_back(msg->header.stamp.toSec() - ptr->back().curvature / float(1000));
  } else {
    time_buffer.push_back(msg->header.stamp.toSec());
  }
  last_timestamp_lidar = msg->header.stamp.toSec();
  if (first_lidar_time < 0.) first_lidar_time = time_buffer.back();
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
  mtx_buffer.lock();
  double preprocess_start_time = omp_get_wtime();
  if (msg->header.stamp.toSec() < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = msg->header.stamp.toSec();

  if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty()) {
    printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
  }

  if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) {
    timediff_set_flg = true;
    timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
    printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);
  if (first_lidar_time < 0.) first_lidar_time = time_buffer.back();

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr& msg_in) {
  // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
  if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
    msg->header.stamp =
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
  }

  double timestamp = msg->header.stamp.toSec();

  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    ROS_WARN("imu loop back, clear buffer");
    imu_buffer.clear();
  }

  imu_buffer.push_back(msg);
  mtx_buffer.unlock();
  sig_buffer.notify_all();
  last_timestamp_imu = timestamp;
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg) {
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void img_cbk(const sensor_msgs::ImageConstPtr& msg) {
  if (!img_en) {
    return;
  }
  if (msg->header.stamp.toSec() < last_timestamp_img) {
    ROS_ERROR("img loop back, clear buffer");
    img_buffer.clear();
    img_time_buffer.clear();
  }
  mtx_buffer.lock();

  img_buffer.push_back(getImageFromMsg(msg));
  img_time_buffer.push_back(msg->header.stamp.toSec());
  last_timestamp_img = msg->header.stamp.toSec();

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup& meas) {
  meas.clear();

  assert(img_buffer.size() == img_time_buffer.size());

  if ((lidar_en && lidar_buffer.empty() ||
       img_en && img_buffer.empty()) ||
      imu_buffer.empty()) {
    return false;
  }

  if (img_en && (img_buffer.empty() ||
                 img_time_buffer.back() < time_buffer.front())) {
    return false;
  }

  if (img_en && flg_first_img && imu_buffer.empty()) {
    return false;
  }
  flg_first_img = false;

  // std::cout << "1" << std::endl;

  if (!lidar_pushed) {  // If not in lidar scan, need to generate new meas
    if (lidar_buffer.empty()) {
      return false;
    }
    meas.lidar = lidar_buffer.front();  // push the firsrt lidar topic
    if (meas.lidar->points.size() <= 1) {
      mtx_buffer.lock();
      if (img_buffer.size() > 0) {
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        img_buffer.pop_front();
        img_time_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      ROS_ERROR("empty pointcloud");
      return false;
    }
    sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list);
    meas.lidar_beg_time = time_buffer.front();
    lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
    meas.lidar_end_time = lidar_end_time;
    lidar_pushed = true;  // flag
  }

  while (!flg_EKF_inited && !img_time_buffer.empty() &&
         img_time_buffer.front() < lidar_end_time) {
    mtx_buffer.lock();
    img_buffer.pop_front();
    img_time_buffer.pop_front();
    mtx_buffer.unlock();
  }

  if (last_timestamp_imu <= lidar_end_time) {
    return false;
  }

  if (!img_time_buffer.empty() && last_timestamp_imu <= img_time_buffer.front())
    return false;

  // std::cout << "2" << std::endl;

  if (img_buffer.empty()) {
    if (last_timestamp_imu < lidar_end_time) {
      ROS_ERROR("lidar out sync");
      lidar_pushed = false;
      return false;
    }
    struct ImgIMUsGroup m;  // standard method to keep imu message.
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    m.imus_only = true;
    m.imu.clear();
    mtx_buffer.lock();
    while ((!imu_buffer.empty() && (imu_time < lidar_end_time))) {
      imu_time = imu_buffer.front()->header.stamp.toSec();
      if (imu_time > lidar_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false;  // sync one whole lidar scan.
    meas.img_imus.push_back(m);
    return true;
  }
  // std::cout << "3" << std::endl;

  while (imu_buffer.front()->header.stamp.toSec() <= lidar_end_time) {
    struct ImgIMUsGroup m;
    if (img_buffer.empty() || img_time_buffer.front() > lidar_end_time) {
      if (last_timestamp_imu < lidar_end_time) {
        ROS_ERROR("lidar out sync");
        lidar_pushed = false;
        return false;
      }
      double imu_time = imu_buffer.front()->header.stamp.toSec();
      m.imu.clear();
      m.imus_only = true;
      mtx_buffer.lock();
      while (!imu_buffer.empty() && (imu_time <= lidar_end_time)) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time) break;
        m.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.img_imus.push_back(m);
    } else {
      double img_start_time = img_time_buffer.front();
      if (last_timestamp_imu < img_start_time) {
        ROS_ERROR("img out sync");
        lidar_pushed = false;
        return false;
      }

      if (img_start_time < meas.last_update_time) {
        mtx_buffer.lock();
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        continue;
      }

      double imu_time = imu_buffer.front()->header.stamp.toSec();
      m.imu.clear();
      m.img_offset_time = img_start_time - meas.lidar_beg_time;
      m.img = img_buffer.front();
      m.imus_only = false;
      mtx_buffer.lock();
      while ((!imu_buffer.empty() && (imu_time < img_start_time))) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > img_start_time || imu_time > lidar_end_time) break;
        m.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
      }
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.img_imus.push_back(m);
    }

    if (imu_buffer.empty()) {
      ROS_ERROR("imu buffer empty");
    }
  }
  // std::cout << "4" << std::endl;
  lidar_pushed = false;  // sync one whole lidar scan.

  if (!meas.img_imus.empty()) {
    mtx_buffer.lock();
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return true;
  }
  // std::cout << "5" << std::endl;

  return false;
}

int process_increments = 0;
void map_incremental() {
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size);
  PointNoNeedDownsample.reserve(feats_down_size);
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    pointBodyToWorld(&(scan_down_body->points[i]), &(scan_down_world->points[i]));
    /* decide if need add to map */
    if (!Nearest_Points[i].empty() && flg_EKF_inited) {
      const PointVector& points_near = Nearest_Points[i];
      bool need_add = true;
      BoxPointType Box_of_Point;
      PointType downsample_result, mid_point;
      mid_point.x = floor(scan_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.y = floor(scan_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.z = floor(scan_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      float dist = calc_dist(scan_down_world->points[i], mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
        PointNoNeedDownsample.push_back(scan_down_world->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist) {
          need_add = false;
          break;
        }
      }
      if (need_add) PointToAdd.push_back(scan_down_world->points[i]);
    } else {
      PointToAdd.push_back(scan_down_world->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  ikdtree.Add_Points(PointToAdd, true);
  ikdtree.Add_Points(PointNoNeedDownsample, false);
}

PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher& pubScanFullWorld) {
  if (pubScanFullWorld.getNumSubscribers() > 0) {
    int size = scan_undistort_body->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      pointBodyToWorld(&scan_undistort_body->points[i],
                       &laserCloudWorld->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
    laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubScanFullWorld.publish(laserCloudmsg);
  }

  if (pcd_save_en) {
    int size = scan_undistort_body->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      pointBodyToWorld(&scan_undistort_body->points[i],
                       &laserCloudWorld->points[i]);
    }
    *pcl_wait_save += *laserCloudWorld;
  }
}

void publish_frame_body(const ros::Publisher& pubScanFullBody) {
  if (pubScanFullBody.getNumSubscribers() == 0) return;
  int size = scan_undistort_body->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
    pointBodyLidarToIMU(&scan_undistort_body->points[i],
                        &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
  laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
  laserCloudmsg.header.frame_id = "body";
  pubScanFullBody.publish(laserCloudmsg);
}

void publish_frame_color(const ros::Publisher& pubScanFullColor) {
  if (pubScanFullColor.getNumSubscribers() == 0) return;

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*scan_color_world, laserCloudmsg);
  double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
  laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
  laserCloudmsg.header.frame_id = "camera_init";
  pubScanFullColor.publish(laserCloudmsg);
}

void publish_effect_world(const ros::Publisher& pubScanFullEffect) {
  if (pubScanFullEffect.getNumSubscribers() == 0) return;
  PointCloudXYZI::Ptr laserCloudWorld(
      new PointCloudXYZI(effct_feat_num, 1));
  for (int i = 0; i < effct_feat_num; i++) {
    pointBodyToWorld(&laserCloudOri->points[i],
                     &laserCloudWorld->points[i]);
  }
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
  double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
  laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
  laserCloudmsg.header.frame_id = "camera_init";
  pubScanFullEffect.publish(laserCloudmsg);
}

template <typename T>
void set_posestamp(T& out) {
  out.pose.position.x = state_point.pos(0);
  out.pose.position.y = state_point.pos(1);
  out.pose.position.z = state_point.pos(2);
  out.pose.orientation.x = geoQuat.x;
  out.pose.orientation.y = geoQuat.y;
  out.pose.orientation.z = geoQuat.z;
  out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher& pubOdometry) {
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "body";
  double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
  odomAftMapped.header.stamp = ros::Time().fromSec(publish_time);
  set_posestamp(odomAftMapped.pose);
  pubOdometry.publish(odomAftMapped);
  auto P = kf.get_P();
  for (int i = 0; i < 6; i++) {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                  odomAftMapped.pose.pose.position.y,
                                  odomAftMapped.pose.pose.position.z));
  q.setW(odomAftMapped.pose.pose.orientation.w);
  q.setX(odomAftMapped.pose.pose.orientation.x);
  q.setY(odomAftMapped.pose.pose.orientation.y);
  q.setZ(odomAftMapped.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));

  std::ofstream fout(std::string(ROOT_DIR) + "Log/lio.txt", std::ios::app);
  double timestamp = publish_time;
  fout << std::fixed << std::setprecision(15) << timestamp << " "
       << std::setprecision(15)
       << odomAftMapped.pose.pose.position.x << " "
       << odomAftMapped.pose.pose.position.y << " "
       << odomAftMapped.pose.pose.position.z << " "
       << odomAftMapped.pose.pose.orientation.w << " "
       << odomAftMapped.pose.pose.orientation.x << " "
       << odomAftMapped.pose.pose.orientation.y << " "
       << odomAftMapped.pose.pose.orientation.z << std::endl;
}

void publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose);
  double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
  msg_body_pose.header.stamp = ros::Time().fromSec(publish_time);
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  if (pubPath.getNumSubscribers() == 0) return;
  pubPath.publish(path);
}

void h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data) {
  double match_start = omp_get_wtime();
  laserCloudOri->clear();
  corr_normvect->clear();

/** closest surface search and residual computation **/
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size; i++) {
    PointType& point_body = scan_down_body->points[i];
    PointType& point_world = scan_down_world->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto& points_near = Nearest_Points[i];

    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      point_selected_scan[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                          : true;
    }

    if (!point_selected_scan[i]) continue;

    VF(4)
    pabcd;
    point_selected_scan[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f)) {
      float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9) {
        point_selected_scan[i] = true;
        normvec->points[i].x = pabcd(0);
        normvec->points[i].y = pabcd(1);
        normvec->points[i].z = pabcd(2);
        normvec->points[i].intensity = pd2;
      }
    }
  }

  effct_feat_num = 0;

  for (int i = 0; i < feats_down_size; i++) {
    if (point_selected_scan[i]) {
      laserCloudOri->points[effct_feat_num] = scan_down_body->points[i];
      corr_normvect->points[effct_feat_num] = normvec->points[i];
      effct_feat_num++;
    }
  }

  if (effct_feat_num < 1) {
    ekfom_data.valid = false;
    ROS_WARN("No Effective Points! \n");
    return;
  }

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);  // 23
  ekfom_data.h.resize(effct_feat_num);

  for (int i = 0; i < effct_feat_num; i++) {
    const PointType& laser_p = laserCloudOri->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType& norm_p = corr_normvect->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (extrinsic_est_en) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
}

void readParameters(ros::NodeHandle& nh) {
  nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
  nh.param<int>("img_enable", img_en, 1);
  nh.param<int>("lidar_enable", lidar_en, 1);
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<string>("common/img_topic", img_topic, "/usb_cam/image_raw");
  nh.param<double>("filter_size_surf", voxel_grid_size, 0.5);
  nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
  nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
  nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
  nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
  nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
  nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
  nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
  nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
  nh.param<bool>("preprocess/align_timestamp_en", align_timestamp_en, false);
  nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
  nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
  nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
  nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("mapping/Pcl", cam_extrinT, vector<double>());
  nh.param<vector<double>>("mapping/Rcl", cam_extrinR, vector<double>());
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<double>("mapping/laser_point_cov", LASER_POINT_COV, 0.001);

  nh.param<int>("camera/debug", debug, 0);
  nh.param<double>("camera/cam_fx", cam_fx, 453.483063);
  nh.param<double>("camera/cam_fy", cam_fy, 453.254913);
  nh.param<double>("camera/cam_cx", cam_cx, 318.908851);
  nh.param<double>("camera/cam_cy", cam_cy, 234.238189);
  nh.param<int>("camera/grid_size", grid_size, 40);
  nh.param<int>("camera/patch_size", patch_size, 4);

  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  Lidar_T_wrt_Cam << VEC_FROM_ARRAY(cam_extrinT);
  Lidar_R_wrt_Cam << MAT_FROM_ARRAY(cam_extrinR);

  vpnp.reset(new visual_pnp::VisualPNP(nh));
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "livo");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  readParameters(nh);
  cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
  std::ofstream fout(std::string(ROOT_DIR) + "Log/lio.txt");
  fout << "#timestamp x y z q_w q_x q_y q_z" << std::endl;

  /*** ROS subscribe initialization ***/
  ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA
                                ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk)
                                : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
  ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
  ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);
  image_transport::Publisher img_pub = it.advertise("/rgb_img", 1);
  ros::Publisher pubScanFullWorld = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
  ros::Publisher pubScanFullBody = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
  ros::Publisher pubScanFullColor = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_color", 100000);
  ros::Publisher pubScanFullEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
  ros::Publisher pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/lidar_map", 100000);
  ros::Publisher pubOdometry = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
  ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";

  _featsArray.reset(new PointCloudXYZI());

  memset(point_selected_scan, true, sizeof(point_selected_scan));
  voxel_grid_scan.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);

  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

  double epsi[23] = {0.001};
  fill(epsi, epsi + 23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

  signal(SIGINT, SigHandle);
  ros::Rate rate(5000);
  bool status = ros::ok();

  while (status) {
    if (flg_exit) break;
    ros::spinOnce();
    if (!sync_packages(Measures)) {
      status = ros::ok();
      cv::waitKey(1);
      rate.sleep();
      continue;
    }

    if (flg_first_scan) {
      p_imu->first_lidar_time = Measures.lidar_beg_time;
      flg_first_scan = false;
      continue;
    }

    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    std::string process_step;

    assert(Measures.img_imus.size() > 0);

    cv::Mat last_rgb;
    for (const auto& img_imus : Measures.img_imus) {
      p_imu->Foward(Measures, img_imus, kf);
      double front_imu_time = 0.;
      double back_imu_time = 0.;
      if (!img_imus.imu.empty()) {
        front_imu_time = img_imus.imu.front()->header.stamp.toSec();
        back_imu_time = img_imus.imu.back()->header.stamp.toSec();
      }
      if (!img_imus.imus_only && first_lidar_time > 0.) {
        assert(!img_imus.img.empty());
        ROS_INFO("Image Process: image at %f, imus from %f to %f",
                 Measures.lidar_beg_time + img_imus.img_offset_time,
                 front_imu_time, back_imu_time);
        process_step += "I";

        vpnp->UpdateState(kf, img_imus.img, scan_color_world);
        last_rgb = vpnp->img().clone();
        // publish_odometry(pubOdometry);
      } else {
        assert(img_imus.img.empty());
        ROS_INFO("LiDAR Process: lidar at %f, imus from %f to %f",
                 Measures.lidar_end_time, front_imu_time, back_imu_time);
        process_step += "L";
      }
    }

    process_step += "B";
    p_imu->Backward(Measures, kf, scan_undistort_body);
    lidar_pushed = false;
    ROS_INFO_STREAM("Process Step: " + process_step);

    if (scan_undistort_body == nullptr || scan_undistort_body->empty()) {
      ROS_WARN("No point, skip this scan!\n");
      continue;
    }

    flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

    // lasermap_fov_segment();

    voxel_grid_scan.setInputCloud(scan_undistort_body);
    voxel_grid_scan.filter(*scan_down_body);
    feats_down_size = scan_down_body->points.size();

    if (ikdtree.Root_Node == nullptr) {
      if (feats_down_size > 5) {
        ikdtree.set_downsample_param(filter_size_map_min);
        scan_down_world->resize(feats_down_size);
        for (int i = 0; i < feats_down_size; i++) {
          pointBodyToWorld(&(scan_down_body->points[i]), &(scan_down_world->points[i]));
        }
        ikdtree.Build(scan_down_world->points);
      }
      continue;
    }

    if (feats_down_size < 5) {
      ROS_WARN("No point, skip this scan!\n");
      continue;
    }

    normvec->resize(feats_down_size);
    scan_down_world->resize(feats_down_size);

    if (0)  // If you need to see map point, change to "if(1)"
    {
      PointVector().swap(ikdtree.PCL_Storage);
      ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
      featsFromMap->clear();
      featsFromMap->points = ikdtree.PCL_Storage;
    }

    Nearest_Points.resize(feats_down_size);

    /*** iterated state estimation ***/
    double solve_H_time = 0;
    kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    Measures.last_update_time = lidar_end_time;
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];

    map_incremental();

    scan_color_world->resize(scan_undistort_body->size());
    for (int i = 0; i < scan_undistort_body->size(); i++) {
      PointType pt;
      pointBodyToWorld(&scan_undistort_body->points[i],
                       &pt);
      scan_color_world->points[i].x = pt.x;
      scan_color_world->points[i].y = pt.y;
      scan_color_world->points[i].z = pt.z;
      scan_color_world->points[i].r = 0u;
      scan_color_world->points[i].g = 0u;
      scan_color_world->points[i].b = 0u;
    }
    vpnp->RenderCloud(scan_color_world);

    publish_odometry(pubOdometry);
    publish_path(pubPath);
    publish_frame_world(pubScanFullWorld);
    publish_frame_body(pubScanFullBody);
    publish_frame_color(pubScanFullColor);
    publish_effect_world(pubScanFullEffect);
    vpnp->Publish();
  }         

  if (pcl_wait_save->size() > 0 && pcd_save_en) {
    string file_name = string("scans.pcd");
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    pcl::PCDWriter pcd_writer;
    cout << "current scan saved to /PCD/" << file_name << endl;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
  }

  return 0;
}
// #include <csignal>
// #include <cmath>
// #include <condition_variable>
// #include <fstream>
// #include <iostream>
// #include <math.h>
// #include <mutex>
// #include <omp.h>
// #include <string>
// #include <thread>
// #include <unistd.h>
// #include <vector>
// #include <deque>
// #include <memory>
// #include <iomanip>
// #include <algorithm>

// #include <Eigen/Core>
// #include <ros/ros.h>
// #include <image_transport/image_transport.h>
// #include <nav_msgs/Odometry.h>
// #include <nav_msgs/Path.h>
// #include <visualization_msgs/Marker.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <sensor_msgs/Imu.h>
// #include <geometry_msgs/Vector3.h>
// #include <tf/transform_datatypes.h>
// #include <tf/transform_broadcaster.h>

// #include <pcl/filters/voxel_grid.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl_conversions/pcl_conversions.h>

// #include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

// #include <livox_ros_driver/CustomMsg.h>

// #include "so3_math.h"
// #include "IMU_Processing.hpp"
// #include "preprocess.h"
// #include "vikit/camera_loader.h"
// #include "ikd-Tree/ikd_Tree.h"
// #include "visual_pnp.h"

// double INIT_TIME = 0.1;
// double LASER_POINT_COV = 0.001;

// bool pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, align_timestamp_en = false;

// float DET_RANGE = 300.0f;
// float MOV_THRESHOLD = 1.5f;
// double time_diff_lidar_to_imu = 0.0;

// int pcd_index = 0;
// int txt_index = -1;

// mutex mtx_buffer;
// condition_variable sig_buffer;

// string root_dir = ROOT_DIR;
// string lid_topic, imu_topic, img_topic, config_file;

// visual_pnp::VisualPNPPtr vpnp = nullptr;

// int img_en = 1, lidar_en = 1, debug = 0;
// double last_timestamp_lidar = 0.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
// double lidar_end_time = 0.0, first_lidar_time = -1.0, first_img_time = -1.0;
// double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
// double voxel_grid_size = 0, filter_size_map_min = 0, cube_len = 0;
// int effct_feat_num = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0;
// bool point_selected_scan[100000] = {0};
// bool lidar_pushed = false, flg_first_scan = true, flg_exit = false, flg_EKF_inited = false;

// bool flg_first_img = true;
// int grid_size, patch_size;
// double cam_fx, cam_fy, cam_cx, cam_cy;

// vector<BoxPointType> cub_needrm;
// vector<PointVector> Nearest_Points;
// vector<double> extrinT(3, 0.0);
// vector<double> extrinR(9, 0.0);
// vector<double> cam_extrinT(3, 0.0);
// vector<double> cam_extrinR(9, 0.0);
// deque<double> time_buffer;
// deque<PointCloudXYZI::Ptr> lidar_buffer;
// deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
// deque<cv::Mat> img_buffer;
// deque<double> img_time_buffer;

// PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
// PointCloudXYZI::Ptr scan_undistort_body(new PointCloudXYZI());
// PointCloudXYZI::Ptr scan_down_body(new PointCloudXYZI());
// PointCloudXYZI::Ptr scan_down_world(new PointCloudXYZI());
// PointCloudXYZRGB::Ptr scan_color_world(new PointCloudXYZRGB());
// PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
// PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
// PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
// PointCloudXYZI::Ptr _featsArray;

// pcl::VoxelGrid<PointType> voxel_grid_scan;

// KD_TREE<PointType> ikdtree;

// V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
// V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
// V3D Lidar_T_wrt_IMU(Zero3d);
// M3D Lidar_R_wrt_IMU(Eye3d);
// V3D Lidar_T_wrt_Cam(Zero3d);
// M3D Lidar_R_wrt_Cam(Eye3d);

// MeasureGroup Measures;
// esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
// state_ikfom state_point;
// vect3 pos_lid;

// nav_msgs::Path path;
// nav_msgs::Odometry odomAftMapped;
// geometry_msgs::Quaternion geoQuat;
// geometry_msgs::PoseStamped msg_body_pose;

// shared_ptr<Preprocess> p_pre(new Preprocess());
// shared_ptr<ImuProcess> p_imu(new ImuProcess());

// void SigHandle(int sig) {
//   flg_exit = true;
//   ROS_WARN("catch sig %d", sig);
//   sig_buffer.notify_all();
// }

// void pointBodyToWorld(PointType const* const pi, PointType* const po) {
//   V3D p_body(pi->x, pi->y, pi->z);
//   V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

//   po->x = p_global(0);
//   po->y = p_global(1);
//   po->z = p_global(2);
//   po->intensity = pi->intensity;
// }

// template <typename T>
// void pointBodyToWorld(const Matrix<T, 3, 1>& pi, Matrix<T, 3, 1>& po) {
//   V3D p_body(pi[0], pi[1], pi[2]);
//   V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

//   po[0] = p_global(0);
//   po[1] = p_global(1);
//   po[2] = p_global(2);
// }

// void pointBodyLidarToIMU(PointType const* const pi, PointType* const po) {
//   V3D p_body_lidar(pi->x, pi->y, pi->z);
//   V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

//   po->x = p_body_imu(0);
//   po->y = p_body_imu(1);
//   po->z = p_body_imu(2);
//   po->intensity = pi->intensity;
// }

// void points_cache_collect() {
//   PointVector points_history;
//   ikdtree.acquire_removed_points(points_history);
//   // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
// }

// BoxPointType LocalMap_Points;
// bool Localmap_Initialized = false;
// void lasermap_fov_segment() {
//   cub_needrm.clear();
//   pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
//   V3D pos_LiD = pos_lid;
//   if (!Localmap_Initialized) {
//     for (int i = 0; i < 3; i++) {
//       LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
//       LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
//     }
//     Localmap_Initialized = true;
//     return;
//   }
//   float dist_to_map_edge[3][2];
//   bool need_move = false;
//   for (int i = 0; i < 3; i++) {
//     dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
//     dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
//     if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
//   }
//   if (!need_move) return;
//   BoxPointType New_LocalMap_Points, tmp_boxpoints;
//   New_LocalMap_Points = LocalMap_Points;
//   float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
//   for (int i = 0; i < 3; i++) {
//     tmp_boxpoints = LocalMap_Points;
//     if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
//       New_LocalMap_Points.vertex_max[i] -= mov_dist;
//       New_LocalMap_Points.vertex_min[i] -= mov_dist;
//       tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
//       cub_needrm.push_back(tmp_boxpoints);
//     } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
//       New_LocalMap_Points.vertex_max[i] += mov_dist;
//       New_LocalMap_Points.vertex_min[i] += mov_dist;
//       tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
//       cub_needrm.push_back(tmp_boxpoints);
//     }
//   }
//   LocalMap_Points = New_LocalMap_Points;

//   points_cache_collect();
//   double delete_begin = omp_get_wtime();
//   if (cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm);
// }

// void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr& msg) {
//   mtx_buffer.lock();
//   double preprocess_start_time = omp_get_wtime();
//   if (msg->header.stamp.toSec() < last_timestamp_lidar) {
//     ROS_ERROR("lidar loop back, clear buffer");
//     lidar_buffer.clear();
//   }

//   PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
//   p_pre->process(msg, ptr);
//   lidar_buffer.push_back(ptr);
//   if (p_pre->lidar_type == 3) {
//     time_buffer.push_back(msg->header.stamp.toSec() - ptr->back().curvature / float(1000));
//   } else {
//     time_buffer.push_back(msg->header.stamp.toSec());
//   }
//   last_timestamp_lidar = msg->header.stamp.toSec();
//   if (first_lidar_time < 0.) first_lidar_time = time_buffer.back();
//   mtx_buffer.unlock();
//   sig_buffer.notify_all();
// }

// double timediff_lidar_wrt_imu = 0.0;
// bool timediff_set_flg = false;
// void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
//   mtx_buffer.lock();
//   double preprocess_start_time = omp_get_wtime();
//   if (msg->header.stamp.toSec() < last_timestamp_lidar) {
//     ROS_ERROR("lidar loop back, clear buffer");
//     lidar_buffer.clear();
//   }
//   last_timestamp_lidar = msg->header.stamp.toSec();

//   if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty()) {
//     printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
//   }

//   if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) {
//     timediff_set_flg = true;
//     timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
//     printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
//   }

//   PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
//   p_pre->process(msg, ptr);
//   lidar_buffer.push_back(ptr);
//   time_buffer.push_back(last_timestamp_lidar);
//   if (first_lidar_time < 0.) first_lidar_time = time_buffer.back();

//   mtx_buffer.unlock();
//   sig_buffer.notify_all();
// }

// void imu_cbk(const sensor_msgs::Imu::ConstPtr& msg_in) {
//   // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
//   sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

//   msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
//   if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
//     msg->header.stamp =
//         ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
//   }

//   double timestamp = msg->header.stamp.toSec();

//   mtx_buffer.lock();

//   if (timestamp < last_timestamp_imu) {
//     ROS_WARN("imu loop back, clear buffer");
//     imu_buffer.clear();
//   }

//   imu_buffer.push_back(msg);
//   mtx_buffer.unlock();
//   sig_buffer.notify_all();
//   last_timestamp_imu = timestamp;
// }

// cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg) {
//   cv::Mat img;
//   img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
//   return img;
// }

// void img_cbk(const sensor_msgs::ImageConstPtr& msg) {
//   if (!img_en) {
//     return;
//   }
//   if (msg->header.stamp.toSec() < last_timestamp_img) {
//     ROS_ERROR("img loop back, clear buffer");
//     img_buffer.clear();
//     img_time_buffer.clear();
//   }
//   mtx_buffer.lock();

//   img_buffer.push_back(getImageFromMsg(msg));
//   img_time_buffer.push_back(msg->header.stamp.toSec());
//   last_timestamp_img = msg->header.stamp.toSec();

//   mtx_buffer.unlock();
//   sig_buffer.notify_all();
// }

// bool sync_packages(MeasureGroup& meas) {
//   meas.clear();

//   assert(img_buffer.size() == img_time_buffer.size());

//   if ((lidar_en && lidar_buffer.empty() ||
//        img_en && img_buffer.empty()) ||
//       imu_buffer.empty()) {
//     return false;
//   }

//   if (img_en && (img_buffer.empty() ||
//                  img_time_buffer.back() < time_buffer.front())) {
//     return false;
//   }

//   if (img_en && flg_first_img && imu_buffer.empty()) {
//     return false;
//   }
//   flg_first_img = false;

//   // std::cout << "1" << std::endl;

//   if (!lidar_pushed) {  // If not in lidar scan, need to generate new meas
//     if (lidar_buffer.empty()) {
//       return false;
//     }
//     meas.lidar = lidar_buffer.front();  // push the firsrt lidar topic
//     if (meas.lidar->points.size() <= 1) {
//       mtx_buffer.lock();
//       if (img_buffer.size() > 0) {
//         lidar_buffer.pop_front();
//         time_buffer.pop_front();
//         img_buffer.pop_front();
//         img_time_buffer.pop_front();
//       }
//       mtx_buffer.unlock();
//       sig_buffer.notify_all();
//       ROS_ERROR("empty pointcloud");
//       return false;
//     }
//     sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list);
//     meas.lidar_beg_time = time_buffer.front();
//     lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
//     meas.lidar_end_time = lidar_end_time;
//     lidar_pushed = true;  // flag
//   }

//   while (!flg_EKF_inited && !img_time_buffer.empty() &&
//          img_time_buffer.front() < lidar_end_time) {
//     mtx_buffer.lock();
//     img_buffer.pop_front();
//     img_time_buffer.pop_front();
//     mtx_buffer.unlock();
//   }

//   if (last_timestamp_imu <= lidar_end_time) {
//     return false;
//   }

//   if (!img_time_buffer.empty() && last_timestamp_imu <= img_time_buffer.front())
//     return false;

//   // std::cout << "2" << std::endl;

//   if (img_buffer.empty()) {
//     if (last_timestamp_imu < lidar_end_time) {
//       ROS_ERROR("lidar out sync");
//       lidar_pushed = false;
//       return false;
//     }
//     struct ImgIMUsGroup m;  // standard method to keep imu message.
//     double imu_time = imu_buffer.front()->header.stamp.toSec();
//     m.imus_only = true;
//     m.imu.clear();
//     mtx_buffer.lock();
//     while ((!imu_buffer.empty() && (imu_time < lidar_end_time))) {
//       imu_time = imu_buffer.front()->header.stamp.toSec();
//       if (imu_time > lidar_end_time) break;
//       m.imu.push_back(imu_buffer.front());
//       imu_buffer.pop_front();
//     }
//     lidar_buffer.pop_front();
//     time_buffer.pop_front();
//     mtx_buffer.unlock();
//     sig_buffer.notify_all();
//     lidar_pushed = false;  // sync one whole lidar scan.
//     meas.img_imus.push_back(m);
//     return true;
//   }
//   // std::cout << "3" << std::endl;

//   while (imu_buffer.front()->header.stamp.toSec() <= lidar_end_time) {
//     struct ImgIMUsGroup m;
//     if (img_buffer.empty() || img_time_buffer.front() > lidar_end_time) {
//       if (last_timestamp_imu < lidar_end_time) {
//         ROS_ERROR("lidar out sync");
//         lidar_pushed = false;
//         return false;
//       }
//       double imu_time = imu_buffer.front()->header.stamp.toSec();
//       m.imu.clear();
//       m.imus_only = true;
//       mtx_buffer.lock();
//       while (!imu_buffer.empty() && (imu_time <= lidar_end_time)) {
//         imu_time = imu_buffer.front()->header.stamp.toSec();
//         if (imu_time > lidar_end_time) break;
//         m.imu.push_back(imu_buffer.front());
//         imu_buffer.pop_front();
//       }
//       mtx_buffer.unlock();
//       sig_buffer.notify_all();
//       meas.img_imus.push_back(m);
//     } else {
//       double img_start_time = img_time_buffer.front();
//       if (last_timestamp_imu < img_start_time) {
//         ROS_ERROR("img out sync");
//         lidar_pushed = false;
//         return false;
//       }

//       if (img_start_time < meas.last_update_time) {
//         mtx_buffer.lock();
//         img_buffer.pop_front();
//         img_time_buffer.pop_front();
//         mtx_buffer.unlock();
//         sig_buffer.notify_all();
//         continue;
//       }

//       double imu_time = imu_buffer.front()->header.stamp.toSec();
//       m.imu.clear();
//       m.img_offset_time = img_start_time - meas.lidar_beg_time;
//       m.img = img_buffer.front();
//       m.imus_only = false;
//       mtx_buffer.lock();
//       while ((!imu_buffer.empty() && (imu_time < img_start_time))) {
//         imu_time = imu_buffer.front()->header.stamp.toSec();
//         if (imu_time > img_start_time || imu_time > lidar_end_time) break;
//         m.imu.push_back(imu_buffer.front());
//         imu_buffer.pop_front();
//       }
//       img_buffer.pop_front();
//       img_time_buffer.pop_front();
//       mtx_buffer.unlock();
//       sig_buffer.notify_all();
//       meas.img_imus.push_back(m);
//     }

//     if (imu_buffer.empty()) {
//       ROS_ERROR("imu buffer empty");
//     }
//   }
//   // std::cout << "4" << std::endl;
//   lidar_pushed = false;  // sync one whole lidar scan.

//   if (!meas.img_imus.empty()) {
//     mtx_buffer.lock();
//     lidar_buffer.pop_front();
//     time_buffer.pop_front();
//     mtx_buffer.unlock();
//     sig_buffer.notify_all();
//     return true;
//   }
//   // std::cout << "5" << std::endl;

//   return false;
// }

// int process_increments = 0;
// void map_incremental() {
//   PointVector PointToAdd;
//   PointVector PointNoNeedDownsample;
//   PointToAdd.reserve(feats_down_size);
//   PointNoNeedDownsample.reserve(feats_down_size);
//   for (int i = 0; i < feats_down_size; i++) {
//     /* transform to world frame */
//     pointBodyToWorld(&(scan_down_body->points[i]), &(scan_down_world->points[i]));
//     /* decide if need add to map */
//     if (!Nearest_Points[i].empty() && flg_EKF_inited) {
//       const PointVector& points_near = Nearest_Points[i];
//       bool need_add = true;
//       BoxPointType Box_of_Point;
//       PointType downsample_result, mid_point;
//       mid_point.x = floor(scan_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
//       mid_point.y = floor(scan_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
//       mid_point.z = floor(scan_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
//       float dist = calc_dist(scan_down_world->points[i], mid_point);
//       if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
//         PointNoNeedDownsample.push_back(scan_down_world->points[i]);
//         continue;
//       }
//       for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
//         if (points_near.size() < NUM_MATCH_POINTS) break;
//         if (calc_dist(points_near[readd_i], mid_point) < dist) {
//           need_add = false;
//           break;
//         }
//       }
//       if (need_add) PointToAdd.push_back(scan_down_world->points[i]);
//     } else {
//       PointToAdd.push_back(scan_down_world->points[i]);
//     }
//   }

//   double st_time = omp_get_wtime();
//   ikdtree.Add_Points(PointToAdd, true);
//   ikdtree.Add_Points(PointNoNeedDownsample, false);
// }

// PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
// void publish_frame_world(const ros::Publisher& pubScanFullWorld) {
//   if (pubScanFullWorld.getNumSubscribers() > 0) {
//     int size = scan_undistort_body->points.size();
//     PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

//     for (int i = 0; i < size; i++) {
//       pointBodyToWorld(&scan_undistort_body->points[i],
//                        &laserCloudWorld->points[i]);
//     }

//     sensor_msgs::PointCloud2 laserCloudmsg;
//     pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
//     double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
//     laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
//     laserCloudmsg.header.frame_id = "camera_init";
//     pubScanFullWorld.publish(laserCloudmsg);
//   }

//   if (pcd_save_en) {
//     int size = scan_undistort_body->points.size();
//     PointCloudXYZI::Ptr laserCloudWorld(
//         new PointCloudXYZI(size, 1));

//     for (int i = 0; i < size; i++) {
//       pointBodyToWorld(&scan_undistort_body->points[i],
//                        &laserCloudWorld->points[i]);
//     }
//     *pcl_wait_save += *laserCloudWorld;
//             pcd_index ++;
//             string all_points_dir(string(string(ROOT_DIR) + "PCD_scans/scans_") + to_string(pcd_index) + string(".pcd"));
//             pcl::PCDWriter pcd_writer;
//             cout << "current scan saved to /PCD_scans/" << all_points_dir << endl;
//             pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
//             pcl_wait_save->clear();
//   }
// }

// void publish_frame_body(const ros::Publisher& pubScanFullBody) {
//   if (pubScanFullBody.getNumSubscribers() == 0) return;
//   int size = scan_undistort_body->points.size();
//   PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

//   for (int i = 0; i < size; i++) {
//     pointBodyLidarToIMU(&scan_undistort_body->points[i],
//                         &laserCloudIMUBody->points[i]);
//   }

//   sensor_msgs::PointCloud2 laserCloudmsg;
//   pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
//   double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
//   laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
//   laserCloudmsg.header.frame_id = "body";
//   pubScanFullBody.publish(laserCloudmsg);
// }

// void publish_frame_color(const ros::Publisher& pubScanFullColor) {
//   if (pubScanFullColor.getNumSubscribers() == 0) return;

//   sensor_msgs::PointCloud2 laserCloudmsg;
//   pcl::toROSMsg(*scan_color_world, laserCloudmsg);
//   double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
//   laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
//   laserCloudmsg.header.frame_id = "camera_init";
//   pubScanFullColor.publish(laserCloudmsg);
// }

// void publish_effect_world(const ros::Publisher& pubScanFullEffect) {
//   if (pubScanFullEffect.getNumSubscribers() == 0) return;
//   PointCloudXYZI::Ptr laserCloudWorld(
//       new PointCloudXYZI(effct_feat_num, 1));
//   for (int i = 0; i < effct_feat_num; i++) {
//     pointBodyToWorld(&laserCloudOri->points[i],
//                      &laserCloudWorld->points[i]);
//   }
//   sensor_msgs::PointCloud2 laserCloudmsg;
//   pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
//   double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
//   laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
//   laserCloudmsg.header.frame_id = "camera_init";
//   pubScanFullEffect.publish(laserCloudmsg);
// }

// template <typename T>
// void set_posestamp(T& out) {
//   out.pose.position.x = state_point.pos(0);
//   out.pose.position.y = state_point.pos(1);
//   out.pose.position.z = state_point.pos(2);
//   out.pose.orientation.x = geoQuat.x;
//   out.pose.orientation.y = geoQuat.y;
//   out.pose.orientation.z = geoQuat.z;
//   out.pose.orientation.w = geoQuat.w;
// }

// void publish_odometry(const ros::Publisher& pubOdometry) {
//   odomAftMapped.header.frame_id = "camera_init";
//   odomAftMapped.child_frame_id = "body";
//   double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
//   odomAftMapped.header.stamp = ros::Time().fromSec(publish_time);
//   set_posestamp(odomAftMapped.pose);
//   pubOdometry.publish(odomAftMapped);
//   auto P = kf.get_P();
//   for (int i = 0; i < 6; i++) {
//     int k = i < 3 ? i + 3 : i - 3;
//     odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
//     odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
//     odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
//     odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
//     odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
//     odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
//   }

//   static tf::TransformBroadcaster br;
//   tf::Transform transform;
//   tf::Quaternion q;
//   transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
//                                   odomAftMapped.pose.pose.position.y,
//                                   odomAftMapped.pose.pose.position.z));
//   q.setW(odomAftMapped.pose.pose.orientation.w);
//   q.setX(odomAftMapped.pose.pose.orientation.x);
//   q.setY(odomAftMapped.pose.pose.orientation.y);
//   q.setZ(odomAftMapped.pose.pose.orientation.z);
//   transform.setRotation(q);
//   br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
//   txt_index ++;
//   std::ofstream fout(std::string(ROOT_DIR) + "Log/lio.txt", std::ios::app);
//   double timestamp = publish_time;

//   // fout << std::fixed << std::setprecision(15) << timestamp << " "
//   //      << std::setprecision(15)
//   fout << txt_index << " "
//        << odomAftMapped.pose.pose.position.x << " "
//        << odomAftMapped.pose.pose.position.y << " "
//        << odomAftMapped.pose.pose.position.z << " "
//        << odomAftMapped.pose.pose.orientation.x << " "
//        << odomAftMapped.pose.pose.orientation.y << " "
//        << odomAftMapped.pose.pose.orientation.z << " "
//        << odomAftMapped.pose.pose.orientation.w << std::endl;
// }

// void publish_path(const ros::Publisher pubPath) {
//   set_posestamp(msg_body_pose);
//   double publish_time = align_timestamp_en ? lidar_end_time - first_lidar_time : lidar_end_time;
//   msg_body_pose.header.stamp = ros::Time().fromSec(publish_time);
//   msg_body_pose.header.frame_id = "camera_init";
//   path.poses.push_back(msg_body_pose);
//   if (pubPath.getNumSubscribers() == 0) return;
//   pubPath.publish(path);
// }

// void h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data) {
//   double match_start = omp_get_wtime();
//   laserCloudOri->clear();
//   corr_normvect->clear();

// /** closest surface search and residual computation **/
// #ifdef MP_EN
//   omp_set_num_threads(MP_PROC_NUM);
// #pragma omp parallel for
// #endif
//   for (int i = 0; i < feats_down_size; i++) {
//     PointType& point_body = scan_down_body->points[i];
//     PointType& point_world = scan_down_world->points[i];

//     /* transform to world frame */
//     V3D p_body(point_body.x, point_body.y, point_body.z);
//     V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
//     point_world.x = p_global(0);
//     point_world.y = p_global(1);
//     point_world.z = p_global(2);
//     point_world.intensity = point_body.intensity;

//     vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

//     auto& points_near = Nearest_Points[i];

//     if (ekfom_data.converge) {
//       /** Find the closest surfaces in the map **/
//       ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
//       point_selected_scan[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
//                                                                                                                           : true;
//     }

//     if (!point_selected_scan[i]) continue;

//     VF(4)
//     pabcd;
//     point_selected_scan[i] = false;
//     if (esti_plane(pabcd, points_near, 0.1f)) {
//       float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
//       float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

//       if (s > 0.9) {
//         point_selected_scan[i] = true;
//         normvec->points[i].x = pabcd(0);
//         normvec->points[i].y = pabcd(1);
//         normvec->points[i].z = pabcd(2);
//         normvec->points[i].intensity = pd2;
//       }
//     }
//   }

//   effct_feat_num = 0;

//   for (int i = 0; i < feats_down_size; i++) {
//     if (point_selected_scan[i]) {
//       laserCloudOri->points[effct_feat_num] = scan_down_body->points[i];
//       corr_normvect->points[effct_feat_num] = normvec->points[i];
//       effct_feat_num++;
//     }
//   }

//   if (effct_feat_num < 1) {
//     ekfom_data.valid = false;
//     ROS_WARN("No Effective Points! \n");
//     return;
//   }

//   /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
//   ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);  // 23
//   ekfom_data.h.resize(effct_feat_num);

//   for (int i = 0; i < effct_feat_num; i++) {
//     const PointType& laser_p = laserCloudOri->points[i];
//     V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
//     M3D point_be_crossmat;
//     point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
//     V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
//     M3D point_crossmat;
//     point_crossmat << SKEW_SYM_MATRX(point_this);

//     /*** get the normal vector of closest surface/corner ***/
//     const PointType& norm_p = corr_normvect->points[i];
//     V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

//     /*** calculate the Measuremnt Jacobian matrix H ***/
//     V3D C(s.rot.conjugate() * norm_vec);
//     V3D A(point_crossmat * C);
//     if (extrinsic_est_en) {
//       V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
//       ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
//     } else {
//       ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
//     }

//     /*** Measuremnt: distance to the closest surface/corner ***/
//     ekfom_data.h(i) = -norm_p.intensity;
//   }
// }

// void readParameters(ros::NodeHandle& nh) {
//   nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
//   nh.param<int>("img_enable", img_en, 1);
//   nh.param<int>("lidar_enable", lidar_en, 1);
//   nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
//   nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
//   nh.param<string>("common/img_topic", img_topic, "/usb_cam/image_raw");
//   nh.param<double>("filter_size_surf", voxel_grid_size, 0.5);
//   nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
//   nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
//   nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
//   nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
//   nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
//   nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
//   nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
//   nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
//   nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
//   nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
//   nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
//   nh.param<bool>("preprocess/align_timestamp_en", align_timestamp_en, false);
//   nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
//   nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
//   nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
//   nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
//   nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
//   nh.param<vector<double>>("mapping/Pcl", cam_extrinT, vector<double>());
//   nh.param<vector<double>>("mapping/Rcl", cam_extrinR, vector<double>());
//   nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
//   nh.param<double>("mapping/laser_point_cov", LASER_POINT_COV, 0.001);

//   nh.param<int>("camera/debug", debug, 0);
//   nh.param<double>("camera/cam_fx", cam_fx, 453.483063);
//   nh.param<double>("camera/cam_fy", cam_fy, 453.254913);
//   nh.param<double>("camera/cam_cx", cam_cx, 318.908851);
//   nh.param<double>("camera/cam_cy", cam_cy, 234.238189);
//   nh.param<int>("camera/grid_size", grid_size, 40);
//   nh.param<int>("camera/patch_size", patch_size, 4);

//   Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
//   Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
//   Lidar_T_wrt_Cam << VEC_FROM_ARRAY(cam_extrinT);
//   Lidar_R_wrt_Cam << MAT_FROM_ARRAY(cam_extrinR);

//   vpnp.reset(new visual_pnp::VisualPNP(nh));
// }

// int main(int argc, char** argv) {
//   ros::init(argc, argv, "livo");
//   ros::NodeHandle nh;
//   image_transport::ImageTransport it(nh);
//   readParameters(nh);
//   cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
//   std::ofstream fout(std::string(ROOT_DIR) + "Log/lio.txt");
//   fout << "#timestamp x y z q_w q_x q_y q_z" << std::endl;

//   /*** ROS subscribe initialization ***/
//   ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA
//                                 ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk)
//                                 : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
//   ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
//   ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);
//   image_transport::Publisher img_pub = it.advertise("/rgb_img", 1);
//   ros::Publisher pubScanFullWorld = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
//   ros::Publisher pubScanFullBody = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
//   ros::Publisher pubScanFullColor = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_color", 100000);
//   ros::Publisher pubScanFullEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
//   ros::Publisher pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/lidar_map", 100000);
//   ros::Publisher pubOdometry = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
//   ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

//   path.header.stamp = ros::Time::now();
//   path.header.frame_id = "camera_init";

//   _featsArray.reset(new PointCloudXYZI());

//   memset(point_selected_scan, true, sizeof(point_selected_scan));
//   voxel_grid_scan.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);

//   p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
//   p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
//   p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
//   p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
//   p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

//   double epsi[23] = {0.001};
//   fill(epsi, epsi + 23, 0.001);
//   kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

//   signal(SIGINT, SigHandle);
//   ros::Rate rate(5000);
//   bool status = ros::ok();

//   while (status) {
//     if (flg_exit) break;
//     ros::spinOnce();
//     if (!sync_packages(Measures)) {
//       status = ros::ok();
//       cv::waitKey(1);
//       rate.sleep();
//       continue;
//     }

//     if (flg_first_scan) {
//       p_imu->first_lidar_time = Measures.lidar_beg_time;
//       flg_first_scan = false;
//       continue;
//     }

//     state_point = kf.get_x();
//     pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

//     std::string process_step;

//     assert(Measures.img_imus.size() > 0);

//     cv::Mat last_rgb;
//     for (const auto& img_imus : Measures.img_imus) {
//       p_imu->Foward(Measures, img_imus, kf);
//       double front_imu_time = 0.;
//       double back_imu_time = 0.;
//       if (!img_imus.imu.empty()) {
//         front_imu_time = img_imus.imu.front()->header.stamp.toSec();
//         back_imu_time = img_imus.imu.back()->header.stamp.toSec();
//       }
//       if (!img_imus.imus_only && first_lidar_time > 0.) {
//         assert(!img_imus.img.empty());
//         ROS_INFO("Image Process: image at %f, imus from %f to %f",
//                  Measures.lidar_beg_time + img_imus.img_offset_time,
//                  front_imu_time, back_imu_time);
//         process_step += "I";

//         vpnp->UpdateState(kf, img_imus.img, scan_color_world);
//         last_rgb = vpnp->img().clone();
//         // publish_odometry(pubOdometry);
//       } else {
//         assert(img_imus.img.empty());
//         ROS_INFO("LiDAR Process: lidar at %f, imus from %f to %f",
//                  Measures.lidar_end_time, front_imu_time, back_imu_time);
//         process_step += "L";
//       }
//     }

//     process_step += "B";
//     p_imu->Backward(Measures, kf, scan_undistort_body);
//     lidar_pushed = false;
//     ROS_INFO_STREAM("Process Step: " + process_step);

//     if (scan_undistort_body == nullptr || scan_undistort_body->empty()) {
//       ROS_WARN("No point, skip this scan!\n");
//       continue;
//     }

//     flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

//     // lasermap_fov_segment();

//     voxel_grid_scan.setInputCloud(scan_undistort_body);
//     voxel_grid_scan.filter(*scan_down_body);
//     feats_down_size = scan_down_body->points.size();

//     if (ikdtree.Root_Node == nullptr) {
//       if (feats_down_size > 5) {
//         ikdtree.set_downsample_param(filter_size_map_min);
//         scan_down_world->resize(feats_down_size);
//         for (int i = 0; i < feats_down_size; i++) {
//           pointBodyToWorld(&(scan_down_body->points[i]), &(scan_down_world->points[i]));
//         }
//         ikdtree.Build(scan_down_world->points);
//       }
//       continue;
//     }

//     if (feats_down_size < 5) {
//       ROS_WARN("No point, skip this scan!\n");
//       continue;
//     }

//     normvec->resize(feats_down_size);
//     scan_down_world->resize(feats_down_size);

//     if (0)  // If you need to see map point, change to "if(1)"
//     {
//       PointVector().swap(ikdtree.PCL_Storage);
//       ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
//       featsFromMap->clear();
//       featsFromMap->points = ikdtree.PCL_Storage;
//     }

//     Nearest_Points.resize(feats_down_size);

//     /*** iterated state estimation ***/
//     double solve_H_time = 0;
//     kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
//     Measures.last_update_time = lidar_end_time;
//     state_point = kf.get_x();
//     pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
//     geoQuat.x = state_point.rot.coeffs()[0];
//     geoQuat.y = state_point.rot.coeffs()[1];
//     geoQuat.z = state_point.rot.coeffs()[2];
//     geoQuat.w = state_point.rot.coeffs()[3];

//     map_incremental();

//     scan_color_world->resize(scan_undistort_body->size());
//     for (int i = 0; i < scan_undistort_body->size(); i++) {
//       PointType pt;
//       pointBodyToWorld(&scan_undistort_body->points[i],
//                        &pt);
//       scan_color_world->points[i].x = pt.x;
//       scan_color_world->points[i].y = pt.y;
//       scan_color_world->points[i].z = pt.z;
//       scan_color_world->points[i].r = 0u;
//       scan_color_world->points[i].g = 0u;
//       scan_color_world->points[i].b = 0u;
//     }
//     vpnp->RenderCloud(scan_color_world);

//     publish_odometry(pubOdometry);
//     publish_path(pubPath);
//     publish_frame_world(pubScanFullWorld);
//     publish_frame_body(pubScanFullBody);
//     publish_frame_color(pubScanFullColor);
//     publish_effect_world(pubScanFullEffect);
//     vpnp->Publish();
//   }

//   // if (pcl_wait_save->size() > 0 && pcd_save_en) {
//   //   string file_name = string("scans.pcd");
//   //   string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
//   //   pcl::PCDWriter pcd_writer;
//   //   cout << "current scan saved to /PCD/" << file_name << endl;
//   //   pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
//   // }

//   return 0;
// }
