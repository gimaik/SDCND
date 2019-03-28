#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.0900, 0.0000, 0.0000,
              0.0000, 0.0009, 0.0000,
              0.0000, 0.0000, 0.0900;

  H_laser_ << 	1, 0, 0, 0,
				0, 1, 0, 0;
  
   Hj_ << 	0, 0, 0 ,0,
     		0, 0, 0, 0,
     		0, 0, 0, 0;

  KalmanFilter ekf_ = KalmanFilter();
  return;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 0.001, 0.001, 0.001, 0.010;

    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 	1,0,0,0,
            	0,1,0,0,
            	0,0,1000,0,
            	0,0,0,1000;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[3];
      ekf_.x_ <<  rho * cos(phi), rho * sin(phi), rho_dot * cos(phi), rho_dot * sin(phi);
      
      if(ekf_.x_(0) < 0.0001) {
        ekf_.x_(0) = 1.0;
        ekf_.P_(0,0) = 1000;
      }
      if(ekf_.x_(1) < 0.0001) {
        ekf_.x_(1) = 1.0;
        ekf_.P_(1,1) = 1000;
      }
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
       ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) /  1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 	1, 0, dt, 0,
    			0, 1, 0, dt,
				0, 0, 1, 0,
    			0, 0, 0, 1;

  float noise_ax = 15.0;
  float noise_ay = 15.0;
  
  float dt2 = pow(dt, 2);
  float dt3 = pow(dt, 3);
  float dt4 = pow(dt, 4);
  
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4 / 4 * noise_ax, 0, dt3 / 2 * noise_ax, 0,    
    		0, dt4 / 4 * noise_ay, 0, dt3 / 2 * noise_ay,
    		dt3 / 2 * noise_ax, 0, dt2 * noise_ax,    0, 
    		0, dt3 / 2 * noise_ay,  0, dt2 * noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    Tools tools; 
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;

  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);   
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
