#include "ukf.h"
#include <iostream>

using namespace std;
using Eigen::Matrix;
using Eigen::MatrixBase;

const double PI = 4 * atan(1);

double normaliseSingle(double x)
{
    return fmod(x + PI, PI * 2) - PI;
}


template<typename T>
auto normalise(const T& matrix) -> decltype(matrix.unaryExpr(&normaliseSingle))
{
    return matrix.unaryExpr(&normaliseSingle);
}


template<int R, int C>
VectorD<R> applyWeights(VectorD<C> weights, MatrixD<R, C> matrix)
{
    return matrix * weights;
};


template<int N, int M1, int M2>
MatrixD<M1, M2> applyWeightedProduct(VectorD<N> weights, MatrixD<M1, N> a, MatrixD<N, M2> b) {
    MatrixD<M1, N> weighted = a.array().rowwise() * weights.transpose().array();
    return weighted * b;
};


template<int R, int C>
MatrixD<R, R> applyWeightedProduct(VectorD<C> weights, MatrixD<R, C> matrix) {
    MatrixD<C, R> transposed = matrix.transpose();
    return applyWeightedProduct(weights, matrix, transposed);
};

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    // initial state vector
    x_ << 0, 0, 0, 0, 0;

    // initial covariance matrix
    P_ << 10, 0, 0, 0, 0,
          0, 10, 0, 0, 0,
          0, 0, 10, 0, 0,
          0, 0, 0, 1.0, 0,
          0, 0, 0, 0, 1.0;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.2;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.10;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.10;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.0175;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.1;

    lambda_ = 3.0 - N_AUG;
    Xsig_pred_.fill(0);
    weights_.fill(1.0 / (lambda_ + N_AUG) / 2.0);
    weights_(0) = lambda_ / (lambda_ + N_AUG);
    std::cout << "weights\n" << weights_ << std::endl;
    std::cout << "sum " << weights_.sum() << std::endl;

    time_us_ = 0;
}

UKF::~UKF()
{}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    if (time_us_ == 0)
    {
        time_us_ = meas_package.timestamp_;
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            double r = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            x_ << r * cos(phi), r * sin(phi), 0, 0, 0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            x_ << meas_package.raw_measurements_(
                0), meas_package.raw_measurements_(1), 0, 0, 0;

        }
        return;
    }


    double dt = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        UpdateRadar(meas_package);
    }
    else
    {
        UpdateLidar(meas_package);
    }
}

VectorD<UKF::N_X> UKF::ProcessModel(VectorD<N_AUG> x, double dt)
{
    VectorD<N_X> x_updated(5);
    double dt2_2 = dt * dt / 2.0;
    double v = x(2);
    double phi = x(3);
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);
    double phi_dot = x(4);
    double nu_a = x(5);
    double nu_phi_ddot = x(6);

    x_updated <<
              dt2_2 * cos_phi * nu_a,
        dt2_2 * sin_phi * nu_a,
        dt * nu_a,
        dt2_2 * nu_phi_ddot,
        dt * nu_phi_ddot;

    double delta_phi = phi_dot * dt;
    x_updated(3) += delta_phi;
    if(fabs(phi_dot) >= 0.001)
    {
        double v_on_phi_dot = v / phi_dot;
        x_updated(0) += v_on_phi_dot * (sin(phi + delta_phi) - sin(phi));
        x_updated(1) += v_on_phi_dot * (-cos(phi + delta_phi) + cos(phi));
    }
    else
    {
        std::cout << phi_dot << std::endl;
        x_updated(0) += v * cos_phi * dt;
        x_updated(1) += v * sin_phi * dt;
    }
    x_updated += x.head(5);
    return x_updated;
}

VectorD<UKF::N_AUG> UKF::AugmentStateVector(VectorD<N_X> x)
{
    //create augmented mean state
    VectorD<N_AUG> x_aug;
    x_aug.head<N_X>() = x_;
    x_aug(N_X) = 0;
    x_aug(N_X + 1) = 0;

    return x_aug;
}

MatrixD<UKF::N_AUG> UKF::AugmentStateCovariance(MatrixD<N_X> P)
{
    //create augmented covariance matrix
    MatrixD<N_AUG> P_aug;
    P_aug.fill(0.0);
    P_aug.topLeftCorner<N_X, N_X>() = P_;
    P_aug(N_X, N_X) = std_a_ * std_a_;
    P_aug(N_X + 1, N_X + 1) = std_yawdd_ * std_yawdd_;

    return P_aug;
}

MatrixD<UKF::N_AUG, 2 * UKF::N_AUG + 1>
UKF::AugmentedSamplePoints(VectorD<N_AUG> x_aug, MatrixD<N_AUG> P_aug)
{
    //create square root matrix
    MatrixD<N_AUG> A = P_aug.llt().matrixL();
    //create augmented sigma points
    MatrixD<N_AUG, 2 * N_AUG + 1> Xsig_aug;
    Xsig_aug.col(0) = x_aug;
    double scale = sqrt(lambda_ + N_AUG);
    for (size_t i = 0; i < N_AUG; ++i)
    {
        VectorD<N_AUG> col = A.col(i);
        Xsig_aug.col(i + 1) = x_aug + scale * col;
        Xsig_aug.col(i + 1 + N_AUG) = x_aug - scale * col;
    }
    return Xsig_aug;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    auto x_aug = AugmentStateVector(x_);
    auto P_aug = AugmentStateCovariance(P_);
    auto Xsig_aug = AugmentedSamplePoints(x_aug, P_aug);

    for (size_t i = 0; i < Xsig_pred_.cols(); ++i)
    {
        Xsig_pred_.col(i) = ProcessModel(Xsig_aug.col(i), delta_t);
    }

    // predict state mean
    x_  = applyWeights(weights_, Xsig_pred_);
    x_(3) = normaliseSingle(x_(3));

    //predict state covariance matrix
    MatrixD<N_X, 2 * N_AUG + 1> err = Xsig_pred_.colwise() - x_;
    err.row(3) = normalise(err.row(3));

    P_ = applyWeightedProduct(weights_, err);
}

void
UKF::KalmanUpdate(MatrixXd S, MatrixXd Zsig_diff, VectorXd z_diff)
{
    MatrixD<N_X, 2 * N_AUG + 1> stateError = Xsig_pred_.colwise() - x_;
    stateError.row(3) = normalise(stateError.row(3));

    MatrixXd Zsig_diff_t = Zsig_diff.transpose();
    MatrixXd Tc = applyWeightedProduct((VectorXd)weights_, (MatrixXd)stateError, Zsig_diff_t);
    std::cout << "Tc" << Tc << std::endl;

    MatrixXd K = Tc * S.inverse();
    std::cout << "K" << K << std::endl;
    x_ += K * z_diff;
    P_ += -K * S * K.transpose();
}



/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    MatrixD<2, 2 * N_AUG + 1> Zsig_pred;
    Zsig_pred.fill(0.0);
    Zsig_pred.row(0) = Xsig_pred_.row(0);
    Zsig_pred.row(1) = Xsig_pred_.row(1);

    VectorD<2> z_pred = applyWeights(weights_, Zsig_pred);

    MatrixD<2, 2 * N_AUG + 1> Zsig_err = Zsig_pred.colwise() - z_pred;
    MatrixD<2, 2> S = applyWeightedProduct(weights_, Zsig_err);

    VectorD<2> R;
    R <<
      std_laspx_ * std_laspx_,
        std_laspy_ * std_laspy_;
    S += R.asDiagonal();
    VectorD<2> z_err = meas_package.raw_measurements_ - z_pred;
    KalmanUpdate(S, Zsig_err, z_err);
}

VectorD<3> RadarPrediction(VectorD<UKF::N_X> state) {
    auto px = state(0);
    auto py = state(1);
    auto v = state(2);
    auto phi = state(3);
    VectorD<3> prediction;
    double rho = sqrt(px * px + py * py);
    if (fabs(rho) <= 0.001) {
        prediction << rho, atan2(py, px), v;
    }
    else {
        prediction << rho, atan2(py, px), (px * cos(phi) + py * sin(phi)) * v / rho;
    }

    return prediction;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    MatrixD<3, 2 * N_AUG + 1> Zsig_pred;
    for (int i = 0; i < 2 * N_AUG + 1; ++i) {
        Zsig_pred.col(i) = RadarPrediction(Xsig_pred_.col(i));
    }

    VectorD<3> z_pred;
    z_pred = applyWeights(weights_, Zsig_pred);
    z_pred(1) = normaliseSingle(z_pred(1));
    // predict state covariance matrix
    VectorD<3> R;
    R <<
      std_radr_ * std_radr_,
        std_radphi_ * std_radphi_,
        std_radrd_ * std_radrd_;
    MatrixD<3, 2 * N_AUG + 1> Zsig_err = Zsig_pred.colwise() - z_pred;
    Zsig_err.row(1) = normalise(Zsig_err.row(1));

    MatrixD<3, 3> S = applyWeightedProduct(weights_, Zsig_err);
    S += R.asDiagonal();

    VectorD<3> z_err = meas_package.raw_measurements_ - z_pred;
    z_err(1) = normaliseSingle(z_err(1));

    KalmanUpdate(S, Zsig_err, z_err);
}
