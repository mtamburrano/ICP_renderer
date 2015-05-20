#define GL_GLEXT_PROTOTYPES

#include <iostream>
#include <stdlib.h>

#include <boost/format.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>
#include <object_recognition_renderer/renderer2d.h>
#include <iostream>

using namespace cv;
using namespace std;

/** get 3D points out of the image */
float matToVec(const cv::Mat_<cv::Vec3f> &src_ref, const cv::Mat_<cv::Vec3f> &src_mod, std::vector<cv::Vec3f>& pts_ref, std::vector<cv::Vec3f>& pts_mod)
{
  pts_ref.clear();
  pts_mod.clear();
  int px_missing = 0;

  cv::MatConstIterator_<cv::Vec3f> it_ref = src_ref.begin();
  cv::MatConstIterator_<cv::Vec3f> it_mod = src_mod.begin();
  for (; it_ref != src_ref.end(); ++it_ref, ++it_mod)
  {
    if (!cv::checkRange(*it_ref))
      continue;

    pts_ref.push_back(*it_ref);
    if (cv::checkRange(*it_mod))
    {
      pts_mod.push_back(*it_mod);
    }
    else
    {
      pts_mod.push_back(cv::Vec3f(0.0f, 0.0f, 0.0f));
      ++px_missing;
    }
  }

  float ratio = 0.0f;
  if ((src_ref.cols > 0) && (src_ref.rows > 0))
    ratio = float(px_missing) / float(src_ref.cols * src_ref.rows);
  return ratio;
}

/** Computes the centroid of 3D points */
void getMean(const std::vector<cv::Vec3f> &pts, cv::Vec3f& centroid)
{
  centroid = cv::Vec3f(0.0f, 0.0f, 0.0f);
  size_t n_points = 0;
  for (std::vector<cv::Vec3f>::const_iterator it = pts.begin(); it != pts.end(); ++it) {
    if (!cv::checkRange(*it))
      continue;
    centroid += (*it);
    ++n_points;
  }

  if (n_points > 0)
  {
    centroid(0) /= float(n_points);
    centroid(1) /= float(n_points);
    centroid(2) /= float(n_points);
  }
}

/** Transforms the point cloud using the rotation and translation */
void transformPoints(const std::vector<cv::Vec3f> &src, std::vector<cv::Vec3f>& dst, const cv::Matx33f &R, const cv::Vec3f &T)
{
  std::vector<cv::Vec3f>::const_iterator it_src = src.begin();
  std::vector<cv::Vec3f>::iterator it_dst = dst.begin();
  for (; it_src != src.end(); ++it_src, ++it_dst) {
    if (!cv::checkRange(*it_src))
      continue;
    (*it_dst) = R * (*it_src) + T;
  }
}

/** Computes the L2 distance between two vectors of 3D points of the same size */
float getL2distClouds(const std::vector<cv::Vec3f> &model, const std::vector<cv::Vec3f> &ref, float &dist_mean, const float mode)
{
  int nbr_inliers = 0;
  int counter = 0;
  float ratio_inliers = 0.0f;

  float dist_expected = dist_mean * 3.0f;
  dist_mean = 0.0f;

  //use the whole region
  std::vector<cv::Vec3f>::const_iterator it_match = model.begin();
  std::vector<cv::Vec3f>::const_iterator it_ref = ref.begin();
  for(; it_match != model.end(); ++it_match, ++it_ref)
  {
    if (!cv::checkRange(*it_ref))
      continue;

    if (cv::checkRange(*it_match))
    {
      float dist = cv::norm(*it_match - *it_ref);
      if ((dist < dist_expected) || (mode == 0))
        dist_mean += dist;
      if (dist < dist_expected)
        ++nbr_inliers;
    }
    ++counter;
  }

  if (counter > 0)
  {
    dist_mean /= float(nbr_inliers);
    ratio_inliers = float(nbr_inliers) / float(counter);
  }
  else
    dist_mean = std::numeric_limits<float>::max();

  return ratio_inliers;
}

/** Refine the object pose by icp (Iterative Closest Point) alignment of two vectors of 3D points.*/
float icpCloudToCloud(const std::vector<cv::Vec3f> &pts_ref, std::vector<cv::Vec3f> &pts_model, cv::Matx33f& R, cv::Vec3f& T, float &px_inliers_ratio, int mode)
{
  //optimal rotation matrix
  cv::Matx33f R_optimal;
  //optimal transformation
  cv::Vec3f T_optimal;

  //the number of desired iterations defined depending on the mode
  int icp_it_th = 35; //maximal number of iterations
  if (mode == 1)
    icp_it_th = 4; //minimal number of iterations
  else if (mode == 2)
    icp_it_th = 4;

  //desired distance between two point clouds
  const float dist_th = 0.012f;
  //The mean distance between the reference and the model point clouds
  float dist_mean = 0.0f;
  px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, mode);
  //The difference between two previously obtained mean distances between the reference and the model point clouds
  float dist_diff = std::numeric_limits<float>::max();

  //the number of performed iterations
  int iter = 0;
  while (( ((dist_mean > dist_th) && (dist_diff > 0.0001f)) || (mode == 1) ) && (iter < icp_it_th))
  {
    ++iter;

    //subsample points from the match and ref clouds
    if (pts_model.empty() || pts_ref.empty())
      continue;

    //compute centroids of each point subset
    cv::Vec3f m_centroid, r_centroid;
    getMean(pts_model, m_centroid);
    getMean(pts_ref, r_centroid);

    //compute the covariance matrix
    cv::Matx33f covariance (0,0,0, 0,0,0, 0,0,0);
    std::vector<cv::Vec3f>::iterator it_s = pts_model.begin();
    std::vector<cv::Vec3f>::const_iterator it_ref = pts_ref.begin();
    for (; it_s < pts_model.end(); ++it_s, ++it_ref)
      covariance += (*it_s) * (*it_ref).t();

    cv::Mat w, u, vt;
    cv::SVD::compute(covariance, w, u, vt);
    //compute the optimal rotation
    R_optimal = cv::Mat(vt.t() * u.t());

    //compute the optimal translation
    T_optimal = r_centroid - R_optimal * m_centroid;
    if (!cv::checkRange(R_optimal) || !cv::checkRange(T_optimal))
      continue;

    //transform the point cloud
    transformPoints(pts_model, pts_model, R_optimal, T_optimal);

    //compute the distance between the transformed and ref point clouds
    dist_diff = dist_mean;
    px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, mode);
    dist_diff -= dist_mean;

    //update the translation matrix: turn to opposite direction at first and then do translation
    T = R_optimal * T;
    //do translation
    cv::add(T, T_optimal, T);
    //update the rotation matrix
    R = R_optimal * R;
    //std::cout << " it " << iter << "/" << icp_it_th << " : " << std::fixed << dist_mean << " " << d_diff << " " << px_inliers_ratio << " " << pts_model.size() << std::endl;
  }

    //std::cout << " icp " << mode << " " << dist_min << " " << iter << "/" << icp_it_th  << " " << px_inliers_ratio << " " << d_diff << " " << std::endl;
  return dist_mean;
}

Mat_<cv::Vec3f> read_mat(const string& filename) {
  Mat_<cv::Vec3f> m;
  FileStorage fs(filename, cv::FileStorage::READ);
  fs["m"] >> m;
  fs.release();
  return m;
}

void visualize_3d_points(const Mat_<cv::Vec3f>& points_3d, std::string name) {
  Mat points_3d_display;
  points_3d.convertTo(points_3d_display, CV_8UC3, 100);
  imshow(name, points_3d_display);
}


void render_and_perform_icp(std::string file_name, std::string points_3d_ref_filename, 
                            std::string points_3d_model_filename, size_t width, size_t height) {
  Renderer3d renderer = Renderer3d(file_name);

  double near = 0.3, far = 3500;
  double focal_length_x = 530, focal_length_y = 530;

  renderer.set_parameters(width, height, focal_length_x, focal_length_y, near, far);
  RendererIterator renderer_iterator = RendererIterator(&renderer, 1);
  
  Matx33f R_object_ref(-0.7366852281578607, -0.4999999600934258, -0.4552965127480917,
                      0.4253253695228831, -0.8660254268245088, 0.2628655362227054,
                      -0.5257311144056662, 0, 0.8506508069388852);
  Vec3d T_object_ref(-0.36801174, 0.0, 0.5954555);
  
  Matx33f R_object_model(-0.8506508159122358, 0, -0.5257310998864799,
                        -0, -1, 0,
                        -0.5257310998864799, 0, 0.8506508159122358);
  Vec3d T_object_model(-0.368012, 0, 0.595456);
  
  // extract the UP vector from R
  cv::Vec3d UP_ref(-(R_object_ref.inv()(0,1)),
                  -(R_object_ref.inv()(1,1)), 
                  -(R_object_ref.inv()(2,1)));
  cv::Vec3d UP_model(-(R_object_model.inv()(0,1)),
                    -(R_object_model.inv()(1,1)),
                    -(R_object_model.inv()(2,1)));
  
  cv::Rect rect;
  cv::Mat image, depth, mask;
  // render reference and model, just for visualization
  try {
    renderer_iterator.render(image, depth, mask, rect, T_object_ref, UP_ref);
    imwrite("rendered_reference.png", image);
    renderer_iterator.render(image, depth, mask, rect, T_object_model, UP_model);
    imwrite("rendered_model.png", image);
  } catch (...) {
      std::cout << "RENDER EXCEPTION" << endl;
  }
  
  // retrieve the mats of 3d points
  Mat_<cv::Vec3f> points_3d_ref = read_mat(points_3d_ref_filename);
  Mat_<cv::Vec3f> points_3d_model = read_mat(points_3d_model_filename);
  
  // show them --warning-- QT could show them a bit deformed, they are fine if saved with imwrite
  visualize_3d_points(points_3d_ref, "reference 3d points");
  visualize_3d_points(points_3d_model, "model 3d points");
  
  // NOW PEFORM ICP
  
  //initialize the translation based on reference data
  cv::Vec3f T_icp = points_3d_ref(points_3d_ref.rows/ 2.0f, points_3d_ref.cols / 2.0f);
  //add the object's depth
  T_icp(2) += 0.065;
  if (!cv::checkRange(T_icp))
    CV_Assert(false);
      
  //initialize the rotation based on model data
  if (!cv::checkRange(R_object_model))
    CV_Assert(false);
  cv::Matx33f R_icp(R_object_model);
  
  //get the point clouds (for both reference and model)
  std::vector<cv::Vec3f> pts_real_model_temp;
  std::vector<cv::Vec3f> pts_real_ref_temp;
  float px_ratio_missing = matToVec(points_3d_model, points_3d_ref, pts_real_model_temp, pts_real_ref_temp);
  if (px_ratio_missing > 0.25f)
    CV_Assert(false);
    
  //perform the first approximate ICP
  float px_ratio_match_inliers = 0.0f;
  float icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_icp, T_icp, px_ratio_match_inliers, 1);
  
  cout <<"Distance after first ICP: "<<icp_dist<<endl; 
  
  //perform a finer ICP
  icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_icp, T_icp, px_ratio_match_inliers, 2);
  
  cout <<"Distance after second ICP: "<<icp_dist<<endl;  
  
  //perform the final precise icp
  float icp_px_match = 0.0f;
  icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_icp, T_icp, icp_px_match, 0);
  
  // extract the UP vector from updated R
  cv::Vec3d UP_icp(-(R_icp.inv()(0,1)),
                  -(R_icp.inv()(1,1)), 
                  -(R_icp.inv()(2,1)));
  
  // render result
  try {
    renderer_iterator.render(image, depth, mask, rect, T_icp, UP_icp);
    imwrite("rendered_icp_result.png", image);
  } catch (...) {
      std::cout << "RENDER EXCEPTION" << endl;
  }
  
  cout << "------ICP output--------" << endl << endl;
  
  cout << "R: "<< R_icp << endl <<endl;
  cout << "T: "<< T_icp << endl <<endl;
  cout <<"Final distance after last ICP: "<<icp_dist<<endl; 
  cout <<"icp_px_match: "<<icp_px_match<<endl; 
  
  waitKey(0);
}

int main(int argc, char **argv) {
  // Define the display
  size_t width = 640, height = 480;

  // the model name can be specified on the command line.
  std::string obj_file_name(argv[1]), file_ext = obj_file_name.substr(obj_file_name.size() - 3, obj_file_name.npos);
  std::string points_3d_ref_filename(argv[2]);
  std::string points_3d_model_filename(argv[3]);
  render_and_perform_icp(obj_file_name, points_3d_ref_filename, points_3d_model_filename, width, height);

  return 0;
}
