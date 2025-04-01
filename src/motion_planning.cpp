/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-10-30 00:05:14
 * @LastEditTime: 2025-03-06 22:07:49
 * @FilePath: /toy-example-of-iLQR/src/motion_planning.cpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "cilqr_solver.hpp"
#include "cubic_spline.hpp"
#include "global_config.hpp"
#include "matplotlibcpp.h"

#include <fmt/core.h>
#include <getopt.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>

namespace plt = matplotlibcpp;

class motion_planner {
private:
    GlobalConfig* config;
    Outlook outlook_ego;
    Outlook outlook_agent;
    Outlook outlook_steering;
    double delta_t;
    double max_simulation_time;
    double target_velocity;
    std::vector<double> reference_x;
    std::vector<double> reference_y;
    std::vector<double> border_widths;
    std::vector<double> center_line_widths;
    std::vector<std::vector<double>> initial_conditions;
    double wheelbase;
    ReferencePoint reference_point;
    double VEHICLE_WIDTH;
    double VEHICLE_HEIGHT;
    double ACC_MAX;
    double d_safe;
    Eigen::Vector3d obs_attr;
    Eigen::Vector2d vehicle_para;
    size_t vehicle_num;
    bool show_reference_line;
    bool show_obstacle_boundary;
    std::vector<double> visual_x_limit;
    std::vector<double> visual_y_limit;
    std::vector<ReferenceLine> borders;
    std::vector<ReferenceLine> center_lines;
    Eigen::Vector2d road_borders;
    double steer_size;
    std::string save_path = "/root/xzcllwx_ws/GameFormer-Planner/plan/";

    CILQRSolver ilqr_solver;

public:
    motion_planner(std::string config_path):
            config(GlobalConfig::get_instance(config_path)),
            ilqr_solver(config) {
        fmt::print("motion_planner init\n");
        if (config_path.empty()) {
            fmt::print("Usage: ./motion_planning -c <config_path>\n");
            exit(EXIT_FAILURE);
        }

        spdlog::set_level(spdlog::level::debug);
        SPDLOG_INFO("config path: {}", config_path);
        config = GlobalConfig::get_instance(config_path);

        delta_t = config->get_config<double>("delta_t"); // 0.1
        max_simulation_time = config->get_config<double>("max_simulation_time"); // 12
        target_velocity = config->get_config<double>("vehicle/target_velocity"); // 8

        // 车道线
        border_widths = config->get_config<std::vector<double>>("laneline/border");  // [-1.8, 1.8, 5.4]
        center_line_widths =
            config->get_config<std::vector<double>>("laneline/center_line"); // [0, 3.6]

        wheelbase = config->get_config<double>("vehicle/wheelbase");
        std::string reference_point_string = config->get_config<std::string>("vehicle/reference_point"); // rear_center or gravity_center
        reference_point = ReferencePoint::GravityCenter;
        if (reference_point_string == "rear_center") {
            reference_point = ReferencePoint::RearCenter;
        }
        VEHICLE_WIDTH = config->get_config<double>("vehicle/width");
        VEHICLE_HEIGHT = config->get_config<double>("vehicle/length");
        ACC_MAX = config->get_config<double>("vehicle/acc_max");
        d_safe = config->get_config<double>("vehicle/d_safe");
        obs_attr = {VEHICLE_WIDTH, VEHICLE_HEIGHT, d_safe};
        vehicle_para = {VEHICLE_HEIGHT, VEHICLE_WIDTH};

        show_reference_line = config->get_config<bool>("visualization/show_reference_line");
        show_obstacle_boundary = config->get_config<bool>("visualization/show_obstacle_boundary");
        // 可视化范围需要调整
        visual_x_limit = {0, 0};
        visual_y_limit = {0, 0};
        if (config->has_key("visualization/x_lim")) {
            visual_x_limit = config->get_config<std::vector<double>>("visualization/x_lim");
        }
        if (config->has_key("visualization/y_lim")) {
            visual_y_limit = config->get_config<std::vector<double>>("visualization/y_lim");
        }

        outlook_ego;
        outlook_agent;
        outlook_steering;
        steer_size = 5;
        std::filesystem::path source_file_path(__FILE__);
        std::filesystem::path project_path = source_file_path.parent_path().parent_path();
        std::string vehicle_pic_path_ego =
            (project_path / "images" / "materials" / "car_cyan.mat.txt").string();
        std::string vehicle_pic_path_agent =
            (project_path / "images" / "materials" / "car_white.mat.txt").string();
        std::string steering_pic_path =
            (project_path / "images" / "materials" / "steering_wheel.mat.txt").string();
        utils::imread(vehicle_pic_path_ego, outlook_ego);
        utils::imread(vehicle_pic_path_agent, outlook_agent);
        utils::imread(steering_pic_path, outlook_steering);
    }
    ~motion_planner() {}

    bool set_reference_line(std::vector<std::vector<double>> reference_line) {
        SPDLOG_INFO("motion_planner set_reference_line");
        borders.clear();
        center_lines.clear();
        reference_x.clear();
        reference_y.clear();
        reference_x = reference_line[0];
        reference_y = reference_line[1];
        for (double w : border_widths) {
            ReferenceLine reference(reference_x, reference_y, w, 1.0);
            borders.emplace_back(reference);
        } // 边缘线
        for (double w : center_line_widths) {
            ReferenceLine reference(reference_x, reference_y, w, 1.0);
            center_lines.emplace_back(reference);
        } // 中心线
        std::sort(border_widths.begin(), border_widths.end(), std::greater<double>());
        road_borders << border_widths[0], border_widths.back();
        return true;
    }

    std::vector<std::vector<double>> plan(
            std::vector<double> initial_conditions,
            std::vector<std::vector<double>> init_trajectory, 
            std::vector<std::vector<std::vector<double>>> predictions,
            double target_vel = 13,
            int iteration = 0) {
        SPDLOG_INFO("motion_planner plan");


        std::vector<RoutingLine> obs_predictions;
        obs_predictions.reserve(predictions.size());
        for (auto& prediction : predictions) {
            obs_predictions.emplace_back(prediction);
        }
        vehicle_num = obs_predictions.size();

        Eigen::Vector4d ego_state = {
                                        initial_conditions[0], 
                                        initial_conditions[1],
                                        initial_conditions[2], 
                                        initial_conditions[3]
                                    };

        target_velocity = target_vel;
        // ReferenceLine reference_line(center_lines[0].x, center_lines[0].y, 
        //                              -initial_conditions[0], -initial_conditions[1],
        //                              center_lines[0].delta_d, center_lines[0].delta_s);
        auto [new_u, new_x] =
            ilqr_solver.solve(ego_state, center_lines[0], target_velocity,
                              obs_predictions, road_borders, init_trajectory);
        // new_x.col(0) = new_x.col(0).array() + initial_conditions[0];
        // new_x.col(1) = new_x.col(1).array() + initial_conditions[1];
        plt::cla();
        ego_state = new_x.row(1).transpose();
        // Eigen::MatrixX4d boarder = utils::get_boundary(new_x, VEHICLE_WIDTH * 0.7);
        // std::vector<std::vector<double>> closed_curve = utils::get_closed_curve(boarder);
        // plt::fill(closed_curve[0], closed_curve[1], {{"color", "cyan"}, {"alpha", "0.7"}});
        utils::plot_vehicle(outlook_ego, ego_state, vehicle_para, reference_point, wheelbase);
        {
            std::vector<double> traj_x;
            std::vector<double> traj_y;
            traj_x.reserve(new_x.rows());
            traj_y.reserve(new_x.rows());
            for (int i = 0; i < new_x.rows(); ++i) {
                traj_x.push_back(new_x(i, 0));  // 矩阵第一列 - x坐标
                traj_y.push_back(new_x(i, 1));  // 矩阵第二列 - y坐标
            }
            plt::scatter(traj_x, traj_y, 50.0, {{"color", "blue"}, {"marker", "o"}});

            std::vector<double> x, y;
            x.reserve(init_trajectory.size());
            y.reserve(init_trajectory.size());
            for (size_t i = 0; i < init_trajectory.size(); ++i) {
                x.emplace_back(init_trajectory[i][0]);
                y.emplace_back(init_trajectory[i][1]);
            }
            // plt::plot(x, y, "-b");
            plt::scatter(x, y, 20.0, {{"color", "green"}, {"marker", "^"}});
        }

        for (size_t idx = 0; idx < obs_predictions.size(); ++idx) {
            plt::plot(obs_predictions[idx].x, obs_predictions[idx].y, "-r");
            Eigen::Vector3d obs_state = obs_predictions[idx][0];
            utils::plot_vehicle(outlook_agent, obs_state, vehicle_para,
                                reference_point, 0);
        }
        for (size_t i = 0; i < borders.size(); ++i) {
            if (i == 0 || i == borders.size() - 1) {
                plt::plot(borders[i].x, borders[i].y, {{"linewidth", "2"}, {"color", "k"}});
            } else {
                plt::plot(borders[i].x, borders[i].y, "-k");
            }
        }
        for (size_t i = 0; i < center_lines.size(); ++i) {
            plt::plot(center_lines[i].x, center_lines[i].y, "--k");
        }
        if (show_obstacle_boundary) {
            Eigen::Matrix3Xd cur_obstacle_states =
                utils::get_cur_obstacle_states(obs_predictions, 0);
            utils::plot_obstacle_boundary(ego_state, cur_obstacle_states, obs_attr, wheelbase,
                                        reference_point);
        }
        // if (show_reference_line) {
        //     plt::plot(center_lines[0].x, center_lines[0].y, "-r");
        // }

        // defualt figure x-y limit
        double visual_x_min = ego_state.x() - 50;
        double visual_y_min = ego_state.y() - 30;
        double visual_x_max = ego_state.x() + 50;
        double visual_y_max = ego_state.y() + 30;
        // if (hypot(visual_x_limit[0], visual_x_limit[1]) > 1e-3) {
        //     visual_x_min = visual_x_limit[0];
        //     visual_x_max = visual_x_limit[1];
        // }
        // if (hypot(visual_y_limit[0], visual_y_limit[1]) > 1e-3) {
        //     visual_y_min = visual_y_limit[0];
        //     visual_y_max = visual_y_limit[1];
        // }

        std::vector<double> outlook_steer_pos = {visual_x_min + steer_size / 1.5,
                                                visual_y_max - steer_size / 1.5,
                                                new_u.row(0)[1] * 2.5};
        utils::imshow(outlook_steering, outlook_steer_pos, {steer_size, steer_size});
        double acc = new_u.row(0)[0] > 0 ? new_u.row(0)[0] : 0;
        double brake = new_u.row(0)[0] > 0 ? 0 : -new_u.row(0)[0];
        double bar_bottom = visual_y_max - steer_size;
        double bar_left = visual_x_min + steer_size * 1.3;
        std::vector<double> acc_bar_x = {bar_left, bar_left + 1, bar_left + 1, bar_left};
        std::vector<double> acc_bar_y = {bar_bottom, bar_bottom,
                                        bar_bottom + steer_size * (acc / ACC_MAX),
                                        bar_bottom + steer_size * (acc / ACC_MAX)};
        std::vector<double> brake_bar_x = {bar_left + 2, bar_left + 3, bar_left + 3, bar_left + 2};
        std::vector<double> brake_bar_y = {bar_bottom, bar_bottom,
                                        bar_bottom + steer_size * (brake / ACC_MAX),
                                        bar_bottom + steer_size * (brake / ACC_MAX)};
        plt::fill(acc_bar_x, acc_bar_y, {{"color", "red"}});
        plt::fill(brake_bar_x, brake_bar_y, {{"color", "gray"}});
        double text_left = bar_left + 4.5;
        double text_top = visual_y_max - 1.5;
        plt::text(text_left, text_top, fmt::format("x = {:.2f} m", ego_state.x()),
                {{"color", "black"}});
        plt::text(text_left, text_top - 1.5, fmt::format("y = {:.2f} m", ego_state.y()),
                {{"color", "black"}});
        plt::text(text_left, text_top - 3, fmt::format("v = {:.2f} m / s", ego_state.z()),
                {{"color", "black"}});
        plt::text(text_left, text_top - 4.5, fmt::format("yaw = {:.2f} rad", ego_state.w()),
                {{"color", "black"}});
        plt::text(text_left + 20, text_top, fmt::format("acc = {:.2f}", new_u.row(0)[0]),
                {{"color", "black"}});
        plt::text(text_left + 20, text_top - 1.5, fmt::format("steer = {:.2f}", new_u.row(0)[1]),
                {{"color", "black"}});

        plt::xlim(visual_x_min, visual_x_max);
        plt::ylim(visual_y_min, visual_y_max);
       
        if (!save_path.empty()) {
            // SPDLOG_INFO("save_path: {}", save_path);
            std::ostringstream oss;
            oss << save_path << iteration << ".png";
            plt::save(oss.str());
        }
        // plt::pause(delta_t);

        // config->destroy_instance();
        // plt::show();
        int N = init_trajectory.size();
        std::vector<std::vector<double>> result(N, std::vector<double>(4, 0));
        for (int i=0; i<N; i++) {
            result[i][0] = new_x.row(i)[0];
            result[i][1] = new_x.row(i)[1];
            result[i][2] = new_x.row(i)[2];
            result[i][3] = new_x.row(i)[3];
        }

        return result; 
    }
};

namespace py = pybind11;

PYBIND11_MODULE(motion_planning, m)
{
    m.doc() = "pybind11 motion_planning plugin";

    py::class_<motion_planner>(m, "motion_planner")
        .def(py::init<std::string>())
        .def("set_reference_line", &motion_planner::set_reference_line)
        .def("plan", &motion_planner::plan, 
             py::arg("initial_conditions"), 
             py::arg("init_trajectory"), 
             py::arg("predictions"),
             py::arg("target_vel"), 
             py::arg("iteration"),
             py::return_value_policy::copy);
}


// int main() {
//     motion_planner planner("/root/xzcllwx_ws/GameFormer-Planner/iLQR/config/scenario_two_borrow.yaml");
//     std::vector<double> initial_conditions = {0, 0, 1, 0};
//     std::vector<std::vector<double>> reference_line = {{-10, 0}, {0, 1}, {50, 2}, {100, 3}, {150, 4}};
//     // 轨迹十个点
//     // std::vector<std::vector<double>> init_trajectory = {
//     //     {1,2,3,4,5,6,7,8,9,10},
//     //     {0,0,0,0,0,0,0,0,0,0},
//     //     {1,1,1,1,1,1,1,1,1,1},
//     //     {0,0,0,0,0,0,0,0,0,0}
//     // };
//     std::vector<std::vector<double>> init_trajectory = {
//         {1, 0, 1, 0}, 
//         {2, 0, 1, 0}, 
//         {3, 0, 1, 0}, 
//         {4, 0, 1, 0}, 
//         {5, 0, 1, 0}, 
//         {6, 0, 1, 0}, 
//         {7, 0, 1, 0}, 
//         {8, 0, 1, 0}, 
//         {9, 0, 1, 0}, 
//         {10, 0, 1, 0}
//     };
//     std::vector<std::vector<std::vector<double>>> predictions = 
//     {
//         {
//             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
//             {3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4},
//             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}            
//         }
//     };
//     // std::vector<std::vector<std::vector<double>>> predictions = 
//     // {
//     //     {
//     //         {1, 3.4, 0}, 
//     //         {2, 3.4, 0}, 
//     //         {3, 3.4, 0}, 
//     //         {4, 3.4, 0}, 
//     //         {5, 3.4, 0}, 
//     //         {6, 3.4, 0}, 
//     //         {7, 3.4, 0}, 
//     //         {8, 3.4, 0}, 
//     //         {9, 3.4, 0}, 
//     //         {10, 3.4, 0}        
//     //     }
//     // };
//     planner.initialize(reference_line);
//     planner.plan(initial_conditions, init_trajectory, predictions);
//     return 0;
// }