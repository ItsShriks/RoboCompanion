import numpy as np
import tf2_ros
from visualization_msgs.msg import Marker
import ros_numpy
import tf2_geometry_msgs
import open3d as o3d
import math
import copy
import rospy
import operator
import cv2
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2#, Image as ImageMsg
from sklearn.impute import SimpleImputer
from geometry_msgs.msg import PoseStamped, PoseArray

class t2d2t3d:
    def __init__(self):
 #       self.map_img = cv2.imread(map_img_path)
 #       self.map_grey = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
        self.org = (np.array([0,0,0])+ 51.224998)/0.05 #real to px
        self.visualizations = []

    def cloud_msg_to_ndarray(self, cloud_msg, fields=['x', 'y', 'z', 'r', 'g', 'b']):
        """
        extract data from a sensor_msgs/PointCloud2 message into a NumPy array
        :type cloud_msg: PointCloud2
        :type fields: data fields in a PointCloud2 message
        :return: NumPy array of given fields
        """
        assert isinstance(cloud_msg, PointCloud2)
        cloud_record = ros_numpy.numpify(cloud_msg)
        cloud_record = ros_numpy.point_cloud2.split_rgb_field(cloud_record)
        cloud_array = np.zeros((*cloud_record.shape, len(fields)))
        index = 0
        for field in fields:
            cloud_array[:, :, index] = cloud_record[field]
            index += 1
        return cloud_array

    def box_coordinate_frame(self, bbox):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mean_coords = bbox.get_center()
        rot = np.array(bbox.R)
        #rot1 = r.as_matrix()
        T = np.zeros((4,4))
        T[:3,:3] = rot
        T[3,3] = 1
        T[0,3] = mean_coords[0]
        T[1,3] = mean_coords[1]
        T[2,3] = mean_coords[2]
        #print(T)
        mesh.scale(0.15, center=(0, 0, 0))
        mesh.transform(T)
        return mesh

    def show_point(self, point, col=[1,1,1]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([point])
        pcd.colors = o3d.utility.Vector3dVector([np.array(col)])
        return pcd

    def dist_3d(self, p1, p2):
        return np.sqrt(np.sum(((p1[0] - p2[0])**2, (p1[1] - p2[1])**2, (p1[2] - p2[2])**2)))

    def numpy_to_open3d(self, data, remove_outliers=True, voxel=True, depth=6.5, voxel_s = 0.01, reshape=True):
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        if reshape == True:
            #a = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
            imputer.fit(data.reshape(data.shape[0]*data.shape[1], data.shape[2]))
            data_no_nans = imputer.transform(data.reshape(data.shape[0]*data.shape[1], data.shape[2]))
        else:
            imputer.fit(data)
            data_no_nans = imputer.transform(data)
        points = o3d.utility.Vector3dVector(data_no_nans)
        cloud_without_nans = o3d.geometry.PointCloud(points)
        if voxel:
            cloud_without_nans = cloud_without_nans.voxel_down_sample(voxel_size=voxel_s)
        if remove_outliers:
            cloud_without_nans, ind = cloud_without_nans.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0) #remove_radius_outlier(nb_points=16, radius=0.05)
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-math.inf, -math.inf, 0]), max_bound=np.array([math.inf, math.inf, depth]))
        cropped = cloud_without_nans.crop(bbox)
        return cropped

    def get_faces(self, bbox):
        vertices = bbox.get_box_points()
        corners_array = np.asarray(vertices)
        back_face = corners_array[[0, 1, 2, 7]]
        front_face = corners_array[[3, 4, 5, 6]]
        right_face = corners_array[[1,6,0,3]]
        left_face = corners_array[[4,7,2,5]]
        #bottom_face = corners_array[[0,2,3,5]]
        #top_face = corners_array[[1, 4, 6, 7]]
        possible_faces = np.array([front_face, left_face, right_face, back_face]) #bottom_face, top_face
        return possible_faces

    def get_box_voxel_simple(self, bbox, cloud):
        cld = self.cloud_msg_to_ndarray(cloud, fields=['x', 'y', 'z'])
        padd = 0
        data_croped = cld[bbox[0][1]:bbox[1][1]-padd,bbox[0][0]:bbox[1][0]-padd,:]
        whole = self.numpy_to_open3d(cld, remove_outliers=0, voxel=0)
        cropped = self.numpy_to_open3d(data_croped, remove_outliers=1, voxel=0)
        cropped.colors = o3d.utility.Vector3dVector([np.array([1,0.5,1])]*len(np.asarray(cropped.points)))
        return whole, cropped

    def get_box_voxel(self, bbox, cloud):
        rob_coords = np.array([0, 0, 0])
        cld = self.cloud_msg_to_ndarray(cloud, fields=['x', 'y', 'z'])
        padd = 0
        data_croped = cld[bbox[0][1]:bbox[1][1]-padd,bbox[0][0]:bbox[1][0]-padd,:]
        whole = self.numpy_to_open3d(cld, remove_outliers=0, voxel=0)
        cropped = self.numpy_to_open3d(data_croped, remove_outliers=1, voxel=0)
        cropped.colors = o3d.utility.Vector3dVector([np.array([1,0.5,1])]*len(np.asarray(cropped.points)))
        obj_bb = cropped.get_oriented_bounding_box()
        obj_mean = obj_bb.center
        rob_coords[1] = obj_mean[1]
        #obj_pose = self.get_3D_cords(cropped) #open3d frame
        #no_plane, plane, planebox = self.plane_remove(whole,obj_pose,plane_orrientation="vertical")
        #if no_plane == None:
            #no_plane = whole
        '''viz=False
        if viz:
            if plane!=None:
                self.visualizations.extend([plane,planebox])
            o3d.visualization.draw_geometries(self.visualizations)'''
        # obj = no_plane.crop(obj_bb)
        obj = whole.crop(obj_bb)
        obj_points = np.asarray(obj.points)
        labels = obj.cluster_dbscan(eps=0.02, min_points=10)
        #cluster_dists = []
        cluster_size = []
        labels_index = np.unique(labels)
        for ind in labels_index:
            vals = np.where(labels==ind)[0]
            obj_cluster = obj_points[vals]
            #cluster_cen = np.nanmean(obj_cluster,axis=0)
            cluster_size.append(len(obj_cluster))
            #cluster_dists.append(self.dist_3d(rob_coords, cluster_cen))
        cluster_ind = np.argmax(cluster_size)# np.argmin(cluster_dists)
        obj_clustered = obj_points[np.where(labels==labels_index[cluster_ind])[0]]
        obj_clus = self.numpy_to_open3d(obj_clustered, remove_outliers=0, voxel=0, reshape=0)
        return whole, obj_clus

    def mesh_sphere(self, rob_coords, radius=0.1):
        pcd = self.show_point(rob_coords,col=[0,1,0])
        spheres = o3d.geometry.TriangleMesh()
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.compute_vertex_normals()
        for i, p in enumerate(pcd.points):
            si = copy.deepcopy(s)
            trans = np.identity(4)
            trans[:3, 3] = p
            si.transform(trans)
            si.paint_uniform_color(pcd.colors[i])
            spheres += si
        return spheres

    def check_pose_3D(self, whole, rob_coords, tolerance=100, radius=0.215, height=1, viz=False):
        rob_coords[1] = 0
        points = (np.array([[1,1,0],[1,0,0],[0,1,1],[0,0,1],[-1,1,0],[-1,0,0],[0,1,-1],[0,0,-1]])*np.array([radius,height,radius]))+ np.array(rob_coords)
        #print(points)
        cropp = self.numpy_to_open3d(points, remove_outliers=0, voxel=0, reshape=0)
        #o3d.visualization.draw_geometries([whole,cropp])
        robo_box = cropp.get_oriented_bounding_box()
        obs = whole.crop(robo_box)
        obs, ind = obs.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
        obs_points = np.asarray(obs.points).shape[0]
        if (obs_points > tolerance):
            obstacle_found = True
        else:
            obstacle_found = False
        return obstacle_found, robo_box

    def get_plane_angle_from_horizontal(self,plane_eq):
        a, b, c, d = plane_eq
        normal = np.array([a, b, c])
        vertical_unit = np.array([0, 0, 1])
        dot_prod = np.dot(normal, vertical_unit)
        angle = np.arccos(dot_prod / np.linalg.norm(normal))
        return np.degrees(angle)

    def plane_remove(self,whole,object_pose,padd=0.4,plane_orrientation="horizontal"):
        pointsw = np.array(whole.points)
        pointst = []
        for i in pointsw:
            #print(object_pose.pose.position.z-padd,i[1],object_pose.pose.position.z+padd)
            if (object_pose.pose.position.z-padd<i[1]<object_pose.pose.position.z+padd):
                pointst.append(i)
        roia = np.array(pointst)
        roi = self.numpy_to_open3d(roia, remove_outliers=0, voxel=0, reshape=0)
        plane_model, inliers = roi.segment_plane(distance_threshold=0.005,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        angle = self.get_plane_angle_from_horizontal(plane_model)
        # print(angle,"----------------------")
        p_remove=False
        if plane_orrientation=="horizontal":
            if 80<angle<100:
                p_remove = True
                print("horizontal plane found")
        elif plane_orrientation=="vertical":
            if -20<angle<20:
                p_remove = True
                print("vertical plane found")

        if p_remove:
            print("plane found")
            inlier_cloud = roi.select_by_index(inliers)
            inlier_bbox = inlier_cloud.get_oriented_bounding_box()
            inliers_indices = inlier_bbox.get_point_indices_within_bounding_box(whole.points)
            #dists = whole.compute_point_cloud_distance(inlier_cloud)
            #dists = np.asarray(dists)
            #ind = np.where(dists < 0.01)[0]
            #cloud_in = whole.select_by_index(ind)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            cloud_out = whole.select_by_index(inliers_indices, invert=True)#ind
            return cloud_out, inlier_cloud, inlier_bbox
        else:
            return None, None, None

    def detect_obstacle(self, whole, obj_clus, rob_coords, no_plane, tolerance=100, viz=False):
        if no_plane == None:
            no_plane = whole
        obb_clus_box = obj_clus.get_oriented_bounding_box()
        #obj_mean = obb_clus_box.center
        rob_coords = [rob_coords[1],rob_coords[2],rob_coords[0]]
        #rob_coords = [rob_coords[0], obj_mean[1], rob_coords[2]]
        sph = self.mesh_sphere(rob_coords, radius=0.02)
        possible_faces = self.get_faces(obb_clus_box)
        distances = []
        for i in possible_faces:
            a = np.mean(i, axis=0)
            distances.append(self.dist_3d(rob_coords, a))
        close_face = possible_faces[np.argmin(distances)]

        corno = np.array([close_face[0], close_face[1], close_face[2], close_face[3], rob_coords])
        corno = np.concatenate((corno, sph.sample_points_uniformly().points), axis=0)
        #corno = np.expand_dims(corno, axis=0)
        cropp = self.numpy_to_open3d(corno, remove_outliers=0, voxel=0, reshape=0)
        cropp.colors = o3d.utility.Vector3dVector([np.array([1,0,0])]*len(np.asarray(cropp.points)))
        obb_rob = cropp.get_oriented_bounding_box()
        obb_rob.scale(0.97, center=(0, 0, 0))

        #obj = no_plane.crop(obb_clus_box)
        #obj_points = np.asarray(obj.points)
        #whole_points = np.asarray(no_plane.points)
        #maxp = obj_points.max(axis=0)
        #minp = obj_points.min(axis=0)
        #indexes = np.arange(len(whole_points))
        in_inbox =  obb_clus_box.get_point_indices_within_bounding_box(no_plane.points)
        #np.where((whole_points[:,1] >= minp[1])&(whole_points[:,0] >= minp[0])&(whole_points[:,2] >= minp[2])&
                            #(whole_points[:,1] <= maxp[1])&(whole_points[:,0] <= maxp[0])&(whole_points[:,2] <= maxp[2]))[0] #get_point_indices_within_bounding_box #
        #not_inbox = list(set(indexes) - set(in_inbox))
        pcd = no_plane.select_by_index(in_inbox, invert=True)
        obstacle = pcd.crop(obb_rob)
        obstacle1, ind = obstacle.remove_radius_outlier(nb_points=16, radius=0.05)
        obs_points = np.asarray(obstacle1.points).shape[0]
        #xyz_coords = np.asarray(obstacle1.points)  # Get all remaining obstacle points as (x, y, z) THESE ARE THE OBJECT COORDINATES
        #print(obs_points)

        if (obs_points > tolerance):
            obstacle_found = 1
            sph.paint_uniform_color(np.array([1,0,0]))
            obb_rob.color = [1,0,0]
        else:
            try:
                obstacle_found, robo_box = self.check_pose_3D(whole, rob_coords, tolerance=100, viz=viz)
            except:
                obstacle_found = True
            if obstacle_found:
                obstacle_found = 2
                sph.paint_uniform_color(np.array([1,0,0]))
                obb_rob.color = [1,0,0]
                #robo_box.color = [1,0,0]
            else:
                obstacle_found = 0
                sph.paint_uniform_color(np.array([0,1,0]))
                obb_rob.color = [0,1,0]
                #robo_box.color = [0,1,0]
            #if viz:
                #self.visualizations.extend([robo_box])
        if viz:
            #arm_box_mesh = box_coordinate_frame(obb_rob)
            self.visualizations.extend([sph, obb_rob]) #arm_box_mesh
        return obstacle_found

    def get_y_rot(self, angle):
        y_r = R.from_euler("xyz", [0,0,angle])
        y_quat = y_r.as_quat()
        return y_quat

    def get_3D_cords(self, obj_clus):
        # print('cloud array')
        mean_coords = obj_clus.get_center() #np.nanmean(cld_arr, axis=0)
        aabb = obj_clus.get_oriented_bounding_box() #get_axis_aligned_bounding_box
        r = R.from_matrix(np.array(aabb.R))
        quat = r.as_quat() #(x, y, z, w) #quaternions
        mean_ps = self.create_pose(mean_coords[2],mean_coords[0],mean_coords[1],"open3D",
                                   quat[0],quat[1],quat[2],quat[3])
        # print('pse')
        #print("mean: ", mean_ps.pose)
        # print(mean_ps)
        return mean_ps

    def create_pose(self,x,y,z,frame_id,ox=0.0,oy=0.0,oz=0.0,ow=1.0):
        pose_obj = PoseStamped()
        pose_obj.header.frame_id = frame_id
        pose_obj.header.stamp = rospy.Time().now()
        pose_obj.pose.position.x = x
        pose_obj.pose.position.y = y
        pose_obj.pose.position.z = z
        pose_obj.pose.orientation.x = ox
        pose_obj.pose.orientation.y = oy
        pose_obj.pose.orientation.z = oz
        pose_obj.pose.orientation.w = ow
        return pose_obj

#    def get_valid_coords_2D(self,valid_points,base_pose_robot,real_object_pose,viz=False):
#        valid_points_new = []
#        obj_cor_px = self.real_2_px(np.array([real_object_pose.pose.position.x,real_object_pose.pose.position.y,0.0]))
#        if viz:
#            image = self.map_img.copy()
#        grey = self.map_grey.copy()
#        for real_cord in valid_points:
#            real_cord_px = self.real_2_px(np.array([real_cord.pose.position.x,real_cord.pose.position.y,0.0]))
#            if grey[int(real_cord_px[1]),int(real_cord_px[0])]>240:
#                new_quat = self.get_orient(real_cord_px, obj_cor_px)
#                valid_points_new.append(self.create_pose(real_cord.pose.position.x,real_cord.pose.position.y,0.0,"map",
#                                                        new_quat[0],new_quat[1],new_quat[2],new_quat[3]))
#                if viz:
#                    image = cv2.circle(image, (int(real_cord_px[0]),int(real_cord_px[1])), 3, (0,0,255), -1)
#        if viz:
#            image = cv2.circle(image, (int(obj_cor_px[0]),int(obj_cor_px[1])), 3, (255,0,0), -1)
#            base_pose_robot_px = self.real_2_px(np.array([base_pose_robot.pose.position.x,base_pose_robot.pose.position.y,0.0]))
#            image = cv2.circle(image, (int(base_pose_robot_px[0]),int(base_pose_robot_px[1])), 3, (0,255,0), -1)
#            image =cv2.resize(image,(image.shape[0]//2,image.shape[1]//2))
#            cv2.imshow("px_image", image)
#            cv2.waitKey(0)
#            cv2.destroyWindow("px_image")
#        return valid_points_new

    def list2posearray(self, valid_points, frame='map'):
        valid_points_array = PoseArray()
        valid_points_array.header.stamp = rospy.Time.now()
        valid_points_array.header.frame_id = frame
        for valid_point in valid_points:
            valid_points_array.poses.append(valid_point.pose)
        return valid_points_array

    def get_orient(self,valid_point_px, obj_cor_px):
        z_euler=math.atan2(obj_cor_px[0]-valid_point_px[0],obj_cor_px[1]-valid_point_px[1])
        deg = math.degrees(z_euler)
        if deg<0:
            deg = deg+360
        # print("new rotation ------------------------")
        # print(deg)
        z_euler = math.radians(deg-90)
        # print("-----------------------------------")
        new_quat = self.get_y_rot(z_euler) #check inv
        return new_quat

    def real_2_px(self, cords):
        cords_px = self.org + ((cords*np.array([1,-1, 1]))/0.05)
        return cords_px

    def visualize_whole(self, whole, cropped):
        mean_coords = cropped.get_center()
        meanp = self.show_point(mean_coords,col=[0,1,0])
        box = cropped.get_oriented_bounding_box()
        aabb_mesh = self.box_coordinate_frame(box)
        robot = o3d.geometry.TriangleMesh.create_coordinate_frame()
        robot.scale(0.25, center=(0, 0, 0))
        self.visualizations.extend([cropped, whole, box, meanp, aabb_mesh, robot])

    def get_valid_coords_3D(self,object_pose,whole,obj_clus,min_radius=0.5,max_radius=0.9,num_points=10,padding=0.2,viz=False):
        no_plane, plane, planebox = self.plane_remove(whole,object_pose)
        self.visualizations = []
        valid_points=[]
        if viz:
            self.visualize_whole(whole, obj_clus)
        theta = (np.linspace(0, 2*np.pi, num_points+1)-np.pi/2)[:-1]
        for i in range(len(theta)):
            for r in np.arange(min_radius,max_radius,0.1):
                rob_cord_rel = [0,0,0]
                rob_cord_rel[0] = r*np.sin(theta[i])+object_pose.pose.position.x #x
                rob_cord_rel[1] = r*np.cos(theta[i])+object_pose.pose.position.y #y
                rob_cord_rel[2] = object_pose.pose.position.z
                #try:
                obstacle = self.detect_obstacle(whole, obj_clus, rob_cord_rel,no_plane,tolerance=100,viz=viz)
                # print(obstacle)
                if obstacle==0:
                    rob_cord_rel_pad = [0,0,0]
                    rob_cord_rel_pad[0] = (r+padding)*np.sin(theta[i])+object_pose.pose.position.x #x
                    rob_cord_rel_pad[1] = (r+padding)*np.cos(theta[i])+object_pose.pose.position.y #y
                    rob_cord_rel_pad[2] = object_pose.pose.position.z
                    obstacle_pad = self.detect_obstacle(whole, obj_clus, rob_cord_rel_pad,no_plane,tolerance=100,viz=viz)
                    if obstacle_pad==0:
                        valid_points.append(self.create_pose(rob_cord_rel_pad[0],rob_cord_rel_pad[1],0.0,object_pose.header.frame_id,0.0,0.0,0.0,0.0))
                        break
                elif obstacle==1:
                    break
                #except Exception as e:
                    #print(e)
                    #pass
        if viz:
            if plane!=None:
                self.visualizations.extend([plane,planebox])
            o3d.visualization.draw_geometries(self.visualizations)
        return valid_points

    def order_valid(self, valid_points, base_pose_robot, obj_pose):
        distances = []
        new_order = []
        for i, valid_point in enumerate(valid_points.poses):
            distance_rob = ((valid_point.position.x - base_pose_robot.pose.position.x)**2+(valid_point.position.y - base_pose_robot.pose.position.y)**2)**0.5
            distance_obj = ((valid_point.position.x - obj_pose.pose.position.x)**2+(valid_point.position.y - obj_pose.pose.position.y)**2)**0.5
            distances.append((i,distance_obj,distance_rob))
        sortedi = sorted(distances, key = operator.itemgetter(1, 2))
        sorted_index = list(np.array(sortedi)[:,0].astype(int))
        # print(sorted_index)
        new_order = list(np.array(valid_points.poses)[sorted_index])
        '''for i in sorted_index:
            new_order.append(valid_points.poses[i])'''
        #ordered_point = valid_points[idx[0]]
        valid_points.poses = new_order
        #del valid_points.poses[idx]
        return valid_points#, ordered_point

    def plan_grab_poses(self, bbox, ros_trans, viz=False):
        for a in range(5):
            try:
                #bbox = [[x1,y1],[x2,y2]]
                cloud = ros_trans.cloud_data
                whole, obj_clus = self.get_box_voxel(bbox, cloud)
                if viz:
                    mean_coords = obj_clus.get_center()
                    mea = self.show_point(mean_coords,col=[0,1,0])
                    aabb = obj_clus.get_oriented_bounding_box()
                    o3d.visualization.draw_geometries([whole, obj_clus, mea, aabb])
                obj_pose = self.get_3D_cords(obj_clus) #open3d frame
                sus=True
                break
            except Exception as e:
                print(e)
                print("rescan")
                sus=False
        if not sus:
            return False
        #dist = ((obj_pose.pose.position.x**2)+(obj_pose.pose.position.y**2))**0.5
        #print(dist)
        #if dist > 0.5:
        valid_points_3D = self.get_valid_coords_3D(obj_pose,whole,obj_clus,min_radius=0.5,max_radius=0.9,num_points=15,padding=0.3,viz=viz) #open3d frame
        valid_points_head = ros_trans.transform_3D2head(valid_points_3D)
        real_object_head = ros_trans.transform_3D2head([obj_pose])[0]
        valid_points_map = ros_trans.transform_head2map(valid_points_head)
        real_object_pose = ros_trans.transform_head2map([real_object_head])[0]
        base_pose_robot = ros_trans.rob_cur_cor
        if len(valid_points_map)==0:
            # print("cannot pickup object")
            return False
        valid_points = self.get_valid_coords_2D(valid_points_map,base_pose_robot,real_object_pose,viz=viz)
        if len(valid_points)==0:
            # print("cannot pickup object")
            return False
        valid_pose_array = self.list2posearray(valid_points)
        if len(valid_pose_array.poses)>0:
            valid_points_ordered = self.order_valid(valid_pose_array, base_pose_robot, real_object_pose)
            # print(valid_points_ordered)
            ros_trans.visualize_rviz(valid_points_ordered,real_object_pose,base_pose_robot)
            return [real_object_pose, valid_points_ordered]
        else:
            return [real_object_pose, False]
