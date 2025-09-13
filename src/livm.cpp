#include <fstream>
#include <iomanip>
#include <mutex>
#include <queue>
#include <thread>

#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "STD/STDesc.h"

std::mutex laser_mtx;
std::mutex odom_mtx;
std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;

void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  std::unique_lock<std::mutex> lock(laser_mtx);
  laser_buffer.push(msg);
}

void OdomHandler(const nav_msgs::Odometry::ConstPtr& msg) {
  std::unique_lock<std::mutex> lock(odom_mtx);
  odom_buffer.push(msg);
}

bool syncPackages(PointCloud::Ptr& cloud, Eigen::Affine3d& pose, double& timestamp) {
  if (laser_buffer.empty() || odom_buffer.empty())
    return false;

  auto laser_msg = laser_buffer.front();
  double laser_timestamp = laser_msg->header.stamp.toSec();

  auto odom_msg = odom_buffer.front();
  double odom_timestamp = odom_msg->header.stamp.toSec();

  // check if timestamps are matched
  if (abs(odom_timestamp - laser_timestamp) < 1e-3) {
    pcl::fromROSMsg(*laser_msg, *cloud);
    timestamp = laser_msg->header.stamp.toSec();

    Eigen::Quaterniond r(
        odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
    Eigen::Vector3d t(odom_msg->pose.pose.position.x,
                      odom_msg->pose.pose.position.y,
                      odom_msg->pose.pose.position.z);

    pose = Eigen::Affine3d::Identity();
    pose.translate(t);
    pose.rotate(r);

    std::unique_lock<std::mutex> l_lock(laser_mtx);
    std::unique_lock<std::mutex> o_lock(odom_mtx);

    laser_buffer.pop();
    odom_buffer.pop();

  } else if (odom_timestamp < laser_timestamp) {
    ROS_WARN(
        "Current odometry is earlier than laser scan, discard one "
        "odometry data.");
    std::unique_lock<std::mutex> o_lock(odom_mtx);
    odom_buffer.pop();
    return false;
  } else {
    ROS_WARN(
        "Current laser scan is earlier than odometry, discard one laser scan.");
    std::unique_lock<std::mutex> l_lock(laser_mtx);
    laser_buffer.pop();
    return false;
  }

  return true;
}

void update_poses(const gtsam::Values& estimates,
                  std::vector<Eigen::Affine3d>& poses) {
  assert(estimates.size() == poses.size());

  poses.clear();

  for (int i = 0; i < estimates.size(); ++i) {
    auto est = estimates.at<gtsam::Pose3>(i);
    Eigen::Affine3d est_affine3d(est.matrix());
    poses.push_back(est_affine3d);
  }
}

void visualizeLoopClosure(
    const ros::Publisher& publisher,
    const std::vector<std::pair<int, int>>& loop_container,
    const std::vector<Eigen::Affine3d>& key_pose_vec) {
  if (loop_container.empty())
    return;

  if (publisher.getNumSubscribers() < 1)
    return;

  visualization_msgs::MarkerArray markerArray;
  visualization_msgs::Marker markerNode;
  markerNode.header.frame_id = "camera_init";
  markerNode.action = visualization_msgs::Marker::ADD;
  markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
  markerNode.ns = "loop_nodes";
  markerNode.id = 0;
  markerNode.pose.orientation.w = 1;
  markerNode.scale.x = 0.3;
  markerNode.scale.y = 0.3;
  markerNode.scale.z = 0.3;
  markerNode.color.r = 0;
  markerNode.color.g = 0.8;
  markerNode.color.b = 1;
  markerNode.color.a = 1;

  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "camera_init";
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = "loop_edges";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.1;
  markerEdge.color.r = 0.9;
  markerEdge.color.g = 0.9;
  markerEdge.color.b = 0;
  markerEdge.color.a = 1;

  for (auto it = loop_container.begin(); it != loop_container.end(); ++it) {
    int key_cur = it->first;
    int key_pre = it->second;
    geometry_msgs::Point p;
    p.x = key_pose_vec[key_cur].translation().x();
    p.y = key_pose_vec[key_cur].translation().y();
    p.z = key_pose_vec[key_cur].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
    p.x = key_pose_vec[key_pre].translation().x();
    p.y = key_pose_vec[key_pre].translation().y();
    p.z = key_pose_vec[key_pre].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
  }

  markerArray.markers.push_back(markerNode);
  markerArray.markers.push_back(markerEdge);
  publisher.publish(markerArray);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "mapping");
  ros::NodeHandle nh;

  ConfigSetting config_setting;
  read_parameters(nh, config_setting);
  bool pcd_save_en = nh.param<bool>("loop/pcd_save_en", false);
  double keyframe_dist_thresh = nh.param<double>("loop/keyframe_dist_thresh", 1.0);
  double keyframe_time_thresh = nh.param<double>("loop/keyframe_time_thresh", 30.0);
  double max_loop_distance = nh.param<double>("loop/max_loop_distance", 30.0);

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_color", 100, laserCloudHandler);
  ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/Odometry", 100, OdomHandler);

  ros::Publisher pubKeyCloud = nh.advertise<sensor_msgs::PointCloud2>("/key_cloud", 100);
  ros::Publisher pubCurrentSTD = nh.advertise<visualization_msgs::MarkerArray>("/std_current", 10);
  ros::Publisher pubMatchedSTD = nh.advertise<visualization_msgs::MarkerArray>("/std_matched", 10);
  ros::Publisher pubLoopEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_edge", 10);
  ros::Publisher pubCorrectMap = nh.advertise<sensor_msgs::PointCloud2>("/correct_map", 10000);
  ros::Publisher pubCorrectPath = nh.advertise<nav_msgs::Path>("/correct_path", 100000);
  ros::Publisher pubCurrentCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
  ros::Publisher pubCurrentCorner = nh.advertise<sensor_msgs::PointCloud2>("/key_points_current", 100);
  ros::Publisher pubMatchedCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
  ros::Publisher pubMatchedCorner = nh.advertise<sensor_msgs::PointCloud2>("/key_points_matched", 100);

  STDescManager* std_manager = new STDescManager(config_setting);

  gtsam::Values initial;
  gtsam::NonlinearFactorGraph graph;

  gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4).finished());

  double loopNoiseScore = 1e-1;
  gtsam::Vector robustNoiseVector6(6);
  robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
      loopNoiseScore, loopNoiseScore, loopNoiseScore;
  gtsam::noiseModel::Base::shared_ptr robustLoopNoise =
      gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(1),
          gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  gtsam::ISAM2 isam(parameters);

  size_t keyCloudInd = 0;
  size_t allFrameInd = 0;

  std::vector<PointCloud::Ptr> key_cloud_vec;
  std::vector<int> key_to_all_frame_mapping;

  std::vector<double> key_time_vec;
  std::vector<Eigen::Affine3d> key_pose_vec;
  std::vector<Eigen::Affine3d> origin_key_pose_vec;

  std::vector<double> all_time_vec;
  std::vector<Eigen::Affine3d> all_pose_vec;
  std::vector<Eigen::Affine3d> origin_all_pose_vec;

  std::vector<std::pair<int, int>> loop_container;

  PointCloud::Ptr key_cloud(new PointCloud);
  PointCloud::Ptr key_cloud_ds(new PointCloud);
  PointCloud::Ptr key_cloud_body(new PointCloud);

  bool has_loop_flag = false;
  gtsam::Values curr_estimate;

  while (ros::ok()) {
    ros::spinOnce();
    PointCloud::Ptr current_cloud_body(new PointCloud);
    PointCloud::Ptr current_cloud_world(new PointCloud);
    Eigen::Affine3d pose;
    double timestamp;

    if (syncPackages(current_cloud_world, pose, timestamp)) {
      all_time_vec.push_back(timestamp);
      all_pose_vec.push_back(pose);
      origin_all_pose_vec.push_back(pose);

      // accumulate key cloud
      auto origin_estimate_affine3d = pose;
      down_sampling_voxel(*current_cloud_world, config_setting.ds_size_);
      *key_cloud += *current_cloud_world;
      if (pubKeyCloud.getNumSubscribers() > 0) {
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(*key_cloud, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubKeyCloud.publish(pub_cloud);
      }

      // add pose constraint to graph
      initial.insert(allFrameInd, gtsam::Pose3(pose.matrix()));
      if (allFrameInd == 0) {
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            0, gtsam::Pose3(pose.matrix()), odometryNoise));
      } else {
        auto prev_pose = gtsam::Pose3(origin_all_pose_vec[allFrameInd - 1].matrix());
        auto curr_pose = gtsam::Pose3(pose.matrix());
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            allFrameInd - 1, allFrameInd, prev_pose.between(curr_pose),
            odometryNoise));
      }

      bool is_keyframe = false;
      if (keyCloudInd == 0) {
        is_keyframe = true;
      } else {
        Eigen::Affine3d last_pose = origin_key_pose_vec.back();
        double dist_diff = (pose.translation() - last_pose.translation()).norm();
        double time_diff = timestamp - key_time_vec.back();
        if (dist_diff > keyframe_dist_thresh ||
            time_diff > keyframe_time_thresh) {
          is_keyframe = true;
        }
      }

      if (is_keyframe) {
        int query_frame = keyCloudInd;
        ++keyCloudInd;

        key_time_vec.push_back(timestamp);
        key_pose_vec.push_back(pose);
        origin_key_pose_vec.push_back(pose);
        key_to_all_frame_mapping.push_back(allFrameInd);

        Eigen::Affine3d inverse_pose = pose.inverse();
        pcl::transformPointCloud(*key_cloud, *key_cloud_body, inverse_pose);
        key_cloud_vec.push_back(key_cloud_body);
        *key_cloud_ds = *key_cloud;
        down_sampling_voxel(*key_cloud_ds, config_setting.ds_size_);

        // generate STD descriptors
        std::vector<STDesc> stds_vec;
        std_manager->GenerateSTDescs(key_cloud_ds, stds_vec);
        std_manager->AddSTDescs(stds_vec, pose);
        std_manager->key_cloud_vec_.push_back(key_cloud_ds->makeShared());
        key_cloud.reset(new PointCloud());
        key_cloud_ds.reset(new PointCloud());
        key_cloud_body.reset(new PointCloud());

        sensor_msgs::PointCloud2 pub_cloud;
        if (pubCurrentCloud.getNumSubscribers() > 0) {
          pcl::toROSMsg(*std_manager->key_cloud_vec_.back(), pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubCurrentCloud.publish(pub_cloud);
        }
        if (pubCurrentCorner.getNumSubscribers() > 0) {
          pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubCurrentCorner.publish(pub_cloud);
        }
        if (pubCurrentSTD.getNumSubscribers() > 0) {
          publish_std_descs(stds_vec, pubCurrentSTD);
        }

        std::vector<int> loop_candidates;
        std::vector<double> loop_scores;
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
        std::vector<std::pair<STDesc, STDesc>> best_std_pair;
        std_manager->SearchLoop(stds_vec, loop_candidates, loop_scores, best_transform, best_std_pair);

        if (!loop_candidates.empty() &&
            loop_scores.front() > std_manager->config_setting_.planegeo_icp_thresh_) {
          has_loop_flag = true;
          int detec_frame = loop_candidates.front();
          double loop_distance = (key_pose_vec[query_frame].translation() -
                                  key_pose_vec[detec_frame].translation())
                                     .norm();
          if (max_loop_distance > 0. && loop_distance > max_loop_distance) {
            ROS_WARN("[Loop Detection] Loop distance %.2f exceeds threshold %.2f, ignoring loop.",
                     loop_distance, max_loop_distance);
            has_loop_flag = false;
          } else {
            ROS_INFO("[Loop Detection] Loop found between key frames %d and %d, score: %.4f",
                     query_frame, detec_frame, loop_scores.front());
            std_manager->PlaneGeomrtricIcp(
                std_manager->plane_cloud_vec_.back(),
                std_manager->plane_cloud_vec_[detec_frame], best_transform);

            auto delta_T = Eigen::Affine3d::Identity();
            delta_T.translate(best_transform.first);
            delta_T.rotate(best_transform.second);
            Eigen::Affine3d src_pose_refined = delta_T * origin_key_pose_vec[query_frame];
            Eigen::Affine3d tar_pose = origin_key_pose_vec[detec_frame];

            loop_container.push_back({detec_frame, query_frame});

            int tar_all_frame_idx = key_to_all_frame_mapping[detec_frame];
            int src_all_frame_idx = key_to_all_frame_mapping[query_frame];

            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                tar_all_frame_idx, src_all_frame_idx,
                gtsam::Pose3(tar_pose.matrix())
                    .between(gtsam::Pose3(src_pose_refined.matrix())),
                robustLoopNoise));

            if (pubMatchedCloud.getNumSubscribers() > 0) {
              pcl::toROSMsg(*std_manager->key_cloud_vec_[detec_frame], pub_cloud);
              pub_cloud.header.frame_id = "camera_init";
              pubMatchedCloud.publish(pub_cloud);
            }

            if (pubMatchedCorner.getNumSubscribers() > 0) {
              pcl::toROSMsg(*std_manager->corner_cloud_vec_[detec_frame], pub_cloud);
              pub_cloud.header.frame_id = "camera_init";
              pubMatchedCorner.publish(pub_cloud);
            }
            if (pubMatchedSTD.getNumSubscribers() > 0) {
              publish_std_pairs(best_std_pair, pubMatchedSTD);
            }
          }
        }
      }

      ++allFrameInd;

      isam.update(graph, initial);
      isam.update();

      if (has_loop_flag) {
        isam.update();
        isam.update();
        isam.update();
        isam.update();
        isam.update();
      }

      graph.resize(0);
      initial.clear();

      curr_estimate = isam.calculateEstimate();

      update_poses(curr_estimate, all_pose_vec);

      key_pose_vec.clear();
      for (int i = 0; i < key_to_all_frame_mapping.size(); ++i) {
        int all_frame_idx = key_to_all_frame_mapping[i];
        key_pose_vec.push_back(all_pose_vec[all_frame_idx]);
      }

      if (pubCorrectMap.getNumSubscribers() > 0) {
        PointCloud full_map;
        for (int i = 0; i < key_pose_vec.size(); ++i) {
          PointCloud correct_cloud;
          pcl::transformPointCloud(*key_cloud_vec[i], correct_cloud, key_pose_vec[i]);
          full_map += correct_cloud;
        }
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(full_map, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCorrectMap.publish(pub_cloud);
      }

      if (pubCorrectPath.getNumSubscribers() > 0) {
        nav_msgs::Path correct_path;
        for (int i = 0; i < key_pose_vec.size(); i += 1) {
          geometry_msgs::PoseStamped msg_pose;
          msg_pose.pose.position.x = key_pose_vec[i].translation()[0];
          msg_pose.pose.position.y = key_pose_vec[i].translation()[1];
          msg_pose.pose.position.z = key_pose_vec[i].translation()[2];
          Eigen::Quaterniond pose_q(key_pose_vec[i].rotation());
          msg_pose.header.frame_id = "camera_init";
          msg_pose.pose.orientation.x = pose_q.x();
          msg_pose.pose.orientation.y = pose_q.y();
          msg_pose.pose.orientation.z = pose_q.z();
          msg_pose.pose.orientation.w = pose_q.w();
          correct_path.poses.push_back(msg_pose);
        }
        correct_path.header.stamp = ros::Time::now();
        correct_path.header.frame_id = "camera_init";
        pubCorrectPath.publish(correct_path);
      }

      //visualizeLoopClosure(pubLoopEdge, loop_container, key_pose_vec);

      has_loop_flag = false;
    }
  }

  std::ofstream fout;
  fout.open(std::string(ROOT_DIR) + "/Log/keyloop.txt");
  fout << "# timestamp x y z qx qy qz qw" << std::endl;
  fout.close();
  fout.open(std::string(ROOT_DIR) + "/Log/keylio.txt");
  fout << "# timestamp x y z qx qy qz qw" << std::endl;
  fout.close();
  fout.open(std::string(ROOT_DIR) + "/Log/loop.txt");
  fout << "# timestamp x y z qx qy qz qw" << std::endl;
  fout.close();

  fout.open(std::string(ROOT_DIR) + "/Log/keyloop.txt", std::ios::app);
  for (int i = 0; i < key_pose_vec.size(); i += 1) {
    Eigen::Quaterniond pose_q(key_pose_vec[i].rotation());
    fout << std::fixed << std::setprecision(9) << key_time_vec[i] << " "
         << std::setprecision(6)
         << key_pose_vec[i].translation()[0] << " "
         << key_pose_vec[i].translation()[1] << " "
         << key_pose_vec[i].translation()[2] << " "
         << pose_q.x() << " "
         << pose_q.y() << " "
         << pose_q.z() << " "
         << pose_q.w() << std::endl;
  }
  fout.close();

  fout.open(std::string(ROOT_DIR) + "/Log/keylio.txt", std::ios::app);
  for (int i = 0; i < origin_key_pose_vec.size(); i += 1) {
    Eigen::Quaterniond pose_q(origin_key_pose_vec[i].rotation());
    fout << std::fixed << std::setprecision(9) << key_time_vec[i] << " "
         << std::setprecision(6)
         << origin_key_pose_vec[i].translation()[0] << " "
         << origin_key_pose_vec[i].translation()[1] << " "
         << origin_key_pose_vec[i].translation()[2] << " "
         << pose_q.x() << " "
         << pose_q.y() << " "
         << pose_q.z() << " "
         << pose_q.w() << std::endl;
  }
  fout.close();

  fout.open(std::string(ROOT_DIR) + "/Log/loop.txt", std::ios::app);
  for (int i = 0; i < all_pose_vec.size(); i += 1) {
    Eigen::Quaterniond pose_q(all_pose_vec[i].rotation());
    fout << std::fixed << std::setprecision(9) << all_time_vec[i] << " "
         << std::setprecision(6)
         << all_pose_vec[i].translation()[0] << " "
         << all_pose_vec[i].translation()[1] << " "
         << all_pose_vec[i].translation()[2] << " "
         << pose_q.x() << " "
         << pose_q.y() << " "
         << pose_q.z() << " "
         << pose_q.w() << std::endl;
  }
  fout.close();

  if (pcd_save_en) {
    PointCloud full_map;
    for (int i = 0; i < key_pose_vec.size(); ++i) {
      PointCloud correct_cloud;
      pcl::transformPointCloud(*key_cloud_vec[i], correct_cloud, key_pose_vec[i]);
      full_map += correct_cloud;
    }
    pcl::PCDWriter pcd_writer;
    std::cout << "opt map saved to /PCD/refined.pcd" << std::endl;
    pcd_writer.writeBinary(std::string(ROOT_DIR) + "/PCD/refined.pcd", full_map);
    full_map.clear();
    for (int i = 0; i < origin_key_pose_vec.size(); ++i) {
      PointCloud origin_cloud;
      pcl::transformPointCloud(*key_cloud_vec[i], origin_cloud, origin_key_pose_vec[i]);
      full_map += origin_cloud;
    }
    std::cout << "opt map saved to /PCD/origin.pcd" << std::endl;
    pcd_writer.writeBinary(std::string(ROOT_DIR) + "/PCD/origin.pcd", full_map);
  }

  return 0;
}
