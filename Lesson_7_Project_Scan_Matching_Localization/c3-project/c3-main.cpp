
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>
#include <mutex>

#include <carla/client/Vehicle.h>

// PCL helpers
//#include "render/render.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include "helper.h"
#include <sstream>
#include <chrono> 
#include <ctime> 
#include <cmath>
#include <pcl/registration/icp.h>
#include <pcl/console/time.h>   // TicToc

PointCloudT pclCloud;
cc::Vehicle::Control control;
std::chrono::time_point<std::chrono::system_clock> currentTime;
vector<ControlState> cs;

bool refresh_view = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
{

  	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
	if (event.getKeySym() == "Right" && event.keyDown()){
		cs.push_back(ControlState(0, -0.02, 0));
  	}
	else if (event.getKeySym() == "Left" && event.keyDown()){
		cs.push_back(ControlState(0, 0.02, 0)); 
  	}
  	if (event.getKeySym() == "Up" && event.keyDown()){
		cs.push_back(ControlState(0.1, 0, 0));
  	}
	else if (event.getKeySym() == "Down" && event.keyDown()){
		cs.push_back(ControlState(-0.1, 0, 0)); 
  	}
	if(event.getKeySym() == "a" && event.keyDown()){
		refresh_view = true;
	}
}

void Accuate(ControlState response, cc::Vehicle::Control& state){

	if(response.t > 0){
		if(!state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = false;
			state.throttle = min(response.t, 1.0f);
		}
	}
	else if(response.t < 0){
		response.t = -response.t;
		if(state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = true;
			state.throttle = min(response.t, 1.0f);

		}
	}
	state.steer = min( max(state.steer+response.s, -1.0f), 1.0f);
	state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer){

	BoxQ box;
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    box.cube_length = 4;
    box.cube_width = 2;
    box.cube_height = 2;
	renderBox(viewer, box, num, color, alpha);
}

int main(){

	auto client = cc::Client("localhost", 2000);
	client.SetTimeout(2s);
	auto world = client.GetWorld();

	auto blueprint_library = world.GetBlueprintLibrary();
	auto vehicles = blueprint_library->Filter("vehicle");

	auto map = world.GetMap();
	auto transform = map->GetRecommendedSpawnPoints()[1];
	auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

	// Configure lidar sensor.
	auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
	// You can tune these attributes to change scan density and coverage.
	lidar_bp.SetAttribute("upper_fov", "15");
	lidar_bp.SetAttribute("lower_fov", "-25");
    lidar_bp.SetAttribute("channels", "32");
    lidar_bp.SetAttribute("range", "30");
	lidar_bp.SetAttribute("rotation_frequency", "30");
	lidar_bp.SetAttribute("points_per_second", "500000");

	auto user_offset = cg::Location(0, 0, 0);
	auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
	auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
	auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
	bool new_scan = true;
	std::mutex scan_mutex;
	std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	viewer->setBackgroundColor (0, 0, 0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
	Pose pose(Point(0,0,0), Rotate(0,0,0));

	// Load map
	PointCloudT::Ptr mapCloud(new PointCloudT);
  	pcl::io::loadPCDFile("map.pcd", *mapCloud);
	std::vector<int> mapIndices;
	pcl::removeNaNFromPointCloud(*mapCloud, *mapCloud, mapIndices);
  	cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	renderPointCloud(viewer, mapCloud, "map", Color(0,0,1)); 

	typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr scanCloud (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr alignedCloud (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr transformedScan (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr localMapX (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr localMapCloud (new pcl::PointCloud<PointT>);
	// Keep ICP object alive across frames to avoid repeated setup cost.
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setMaximumIterations(12);
	icp.setMaxCorrespondenceDistance(2.5);
	icp.setTransformationEpsilon(1e-4);
	icp.setEuclideanFitnessEpsilon(1e-3);

	lidar->Listen([&new_scan, &scan_mutex, &lastScanTime, &scanCloud](auto data){

		std::lock_guard<std::mutex> lock(scan_mutex);
		if(new_scan){
			auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
			for (auto detection : *scan){
				if((detection.x*detection.x + detection.y*detection.y + detection.z*detection.z) > 8.0){
					// CARLA lidar axes to project/map convention used by this project.
					const float px = -detection.y;
					const float py = detection.x;
					const float pz = -detection.z;
					if (std::isfinite(px) && std::isfinite(py) && std::isfinite(pz)) {
						pclCloud.points.push_back(PointT(px, py, pz));
					}
				}
			}
			if(pclCloud.points.size() > 3000){ // Batch size for each scan-matching update.
				lastScanTime = std::chrono::system_clock::now();
				*scanCloud = pclCloud;
				new_scan = false;
			}
		}
	});
	
	Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));
	double maxError = 0;

	while (!viewer->wasStopped())
  	{
		while(true){
			bool waiting_scan = true;
			{
				std::lock_guard<std::mutex> lock(scan_mutex);
				waiting_scan = new_scan;
			}
			if(!waiting_scan){
				break;
			}
			std::this_thread::sleep_for(0.1s);
			world.Tick(1s);
		}
		if(refresh_view){
			viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x+1, pose.position.y+1, 0, 0, 0, 1);
			refresh_view = false;
		}
		
		viewer->removeShape("box0");
		viewer->removeShape("boxFill0");
		Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;
		drawCar(truePose, 0,  Color(1,0,0), 0.7, viewer);
		double theta = truePose.rotation.yaw;
		double stheta = control.steer * pi/4 + theta;
		viewer->removeShape("steer");
		renderRay(viewer, Point(truePose.position.x+2*cos(theta), truePose.position.y+2*sin(theta),truePose.position.z),  Point(truePose.position.x+4*cos(stheta), truePose.position.y+4*sin(stheta),truePose.position.z), "steer", Color(0,1,0));


		ControlState accuate(0, 0, 1);
		if(cs.size() > 0){
			accuate = cs.back();
			cs.clear();

			Accuate(accuate, control);
			vehicle->ApplyControl(control);
		}

  		viewer->spinOnce ();
		
		bool has_scan = false;
		{
			std::lock_guard<std::mutex> lock(scan_mutex);
			has_scan = !new_scan;
		}
		if(has_scan){
			PointCloudT::Ptr currentScan(new PointCloudT);
			{
				std::lock_guard<std::mutex> lock(scan_mutex);
				*currentScan = *scanCloud;
				new_scan = true;
				pclCloud.points.clear();
			}

			std::vector<int> scanIndices;
			pcl::removeNaNFromPointCloud(*currentScan, *currentScan, scanIndices);
			if(currentScan->empty()){
				continue;
			}

			// Downsample the raw scan to keep registration fast and reduce noise.
			pcl::VoxelGrid<PointT> voxelFilter;
			voxelFilter.setInputCloud(currentScan);
			voxelFilter.setLeafSize(0.35f, 0.35f, 0.35f);
			voxelFilter.filter(*cloudFiltered);
			currentScan.swap(cloudFiltered);

			// Build the initial registration guess from the last estimated pose.
			// A strong seed improves convergence speed and stability.
			Eigen::Matrix4f initialGuess = Eigen::Matrix4f::Identity();
			const float cy = static_cast<float>(cos(pose.rotation.yaw));
			const float sy = static_cast<float>(sin(pose.rotation.yaw));
			initialGuess(0, 0) = cy;
			initialGuess(0, 1) = -sy;
			initialGuess(1, 0) = sy;
			initialGuess(1, 1) = cy;
			initialGuess(0, 3) = static_cast<float>(pose.position.x);
			initialGuess(1, 3) = static_cast<float>(pose.position.y);
			initialGuess(2, 3) = static_cast<float>(pose.position.z);

			// Crop a local map window around the current estimate so ICP matches faster and more reliably.
			pcl::PassThrough<PointT> passX;
			passX.setInputCloud(mapCloud);
			passX.setFilterFieldName("x");
			passX.setFilterLimits(initialGuess(0, 3) - 45.0f, initialGuess(0, 3) + 45.0f);
			passX.filter(*localMapX);

			pcl::PassThrough<PointT> passY;
			passY.setInputCloud(localMapX);
			passY.setFilterFieldName("y");
			passY.setFilterLimits(initialGuess(1, 3) - 45.0f, initialGuess(1, 3) + 45.0f);
			passY.filter(*localMapCloud);

			if(localMapCloud->size() < 2000){
				*localMapCloud = *mapCloud;
			}

			// Run ICP from the predicted pose and keep only numerically stable results.
			icp.setInputTarget(localMapCloud);
			icp.setInputSource(currentScan);
			icp.align(*alignedCloud, initialGuess);
			const bool icpConverged = icp.hasConverged();
			const double icpFitness = icp.getFitnessScore();
			const bool icpValid = icpConverged && std::isfinite(icpFitness) && icpFitness < 2.5;

			Eigen::Matrix4f matchTransform = initialGuess;
			if (icpValid) {
				matchTransform = icp.getFinalTransformation();
			}

			pose.position.x = matchTransform(0, 3);
			pose.position.y = matchTransform(1, 3);
			pose.position.z = matchTransform(2, 3);
			pose.rotation.yaw = atan2(matchTransform(1, 0), matchTransform(0, 0));

			// Step 3: Render scan points in map frame using the accepted transform.
			pcl::transformPointCloud(*currentScan, *transformedScan, matchTransform);
			viewer->removePointCloud("scan");
			renderPointCloud(viewer, transformedScan, "scan", Color(1,0,0) );

			viewer->removeAllShapes();
			drawCar(pose, 1,  Color(0,1,0), 0.35, viewer);
          
          	double poseError = sqrt( (truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y) );
			if(poseError > maxError)
				maxError = poseError;
			double distDriven = sqrt( (truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y) );
			viewer->removeShape("maxE");
			viewer->addText("Max Error: "+to_string(maxError)+" m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE",0);
			viewer->removeShape("derror");
			viewer->addText("Pose error: "+to_string(poseError)+" m", 200, 150, 32, 1.0, 1.0, 1.0, "derror",0);
			viewer->removeShape("dist");
			viewer->addText("Distance: "+to_string(distDriven)+" m", 200, 200, 32, 1.0, 1.0, 1.0, "dist",0);

			if(maxError > 1.2 || distDriven >= 170.0 ){
				viewer->removeShape("eval");
			if(maxError > 1.2){
				viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval",0);
			}
			else{
				viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval",0);
			}
		}
		}
  	}
	return 0;
}
