/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-10-30 00:05:14
 * @LastEditTime: 2025-02-10 00:43:15
 * @FilePath: /toy-example-of-iLQR/src/motion_planning.cpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

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

std::vector<RoutingLine> get_sub_routing_lines(const std::vector<RoutingLine>& routing_lines,
                                               int start_idx) {
    size_t lines_num = routing_lines.size();
    std::vector<RoutingLine> sub_routing_lines(lines_num);

    for (size_t i = 0; i < lines_num; ++i) {
        std::copy(routing_lines[i].x.begin() + start_idx, routing_lines[i].x.end(),
                  std::back_inserter(sub_routing_lines[i].x));
        std::copy(routing_lines[i].y.begin() + start_idx, routing_lines[i].y.end(),
                  std::back_inserter(sub_routing_lines[i].y));
        std::copy(routing_lines[i].yaw.begin() + start_idx, routing_lines[i].yaw.end(),
                  std::back_inserter(sub_routing_lines[i].yaw));
    }

    return sub_routing_lines;
}

class api_solver {
public:
    api_solver() {
        fmt::print("api_solver init\n");
    }
    ~api_solver() {}
    int demo(std::string config_path) {
        fmt::print("api_solver demo\n");
        if (config_path.empty()) {
            // fmt::print("Usage: %s [-c]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    
        spdlog::set_level(spdlog::level::debug);
        SPDLOG_INFO("config path: {}", config_path);
        GlobalConfig* config = GlobalConfig::get_instance(config_path);
    
        double delta_t = config->get_config<double>("delta_t"); // 0.1
        double max_simulation_time = config->get_config<double>("max_simulation_time"); // 12
        double target_velocity = config->get_config<double>("vehicle/target_velocity"); // 8
        std::vector<double> reference_x =
            config->get_config<std::vector<double>>("laneline/reference/x"); // -10, 0, 50, 100, 150, 200
        std::vector<double> reference_y =
            config->get_config<std::vector<double>>("laneline/reference/y"); // 0, 0, 0, 0, 0, 0
        std::vector<double> border_widths = config->get_config<std::vector<double>>("laneline/border"); // [-1.8, 1.8, 5.4]
        std::vector<double> center_line_widths =
            config->get_config<std::vector<double>>("laneline/center_line"); // [0, 3.6]
        std::vector<std::vector<double>> initial_conditions =
            config->get_config<std::vector<std::vector<double>>>("initial_condition"); // [0, 0, 8.0, 0] 
                                                                                       // [30, 0, 3.0, 0]
                                                                                       // [30, 3.6, 5, 0]
                                                                                       // [15, 3.6, 3.0, 0]
        double wheelbase = config->get_config<double>("vehicle/wheelbase");
        std::string reference_point_string = config->get_config<std::string>("vehicle/reference_point"); // rear_center or gravity_center
        ReferencePoint reference_point = ReferencePoint::GravityCenter;
        if (reference_point_string == "rear_center") {
            reference_point = ReferencePoint::RearCenter;
        }
        double VEHICLE_WIDTH = config->get_config<double>("vehicle/width");
        double VEHICLE_HEIGHT = config->get_config<double>("vehicle/length");
        double ACC_MAX = config->get_config<double>("vehicle/acc_max");
        Eigen::Vector2d vehicle_para = {VEHICLE_HEIGHT, VEHICLE_WIDTH};
        size_t vehicle_num = initial_conditions.size();  // 4
    
        bool show_reference_line = false;
        std::vector<double> visual_x_limit = {0, 0};
        std::vector<double> visual_y_limit = {0, 0};
        if (config->has_key("visualization/x_lim")) {
            visual_x_limit = config->get_config<std::vector<double>>("visualization/x_lim");
        }
        if (config->has_key("visualization/y_lim")) {
            visual_y_limit = config->get_config<std::vector<double>>("visualization/y_lim");
        }
        show_reference_line = config->get_config<bool>("visualization/show_reference_line");
    
        std::vector<ReferenceLine> borders;
        std::vector<ReferenceLine> center_lines;
        for (double w : border_widths) {
            ReferenceLine reference(reference_x, reference_y, w);
            borders.emplace_back(reference);
        } // 边缘线
        for (double w : center_line_widths) {
            ReferenceLine reference(reference_x, reference_y, w);
            center_lines.emplace_back(reference);
        } // 中心线
        std::sort(border_widths.begin(), border_widths.end(), std::greater<double>());
        Eigen::Vector2d road_borders;
        road_borders << border_widths[0], border_widths.back();
    
        Outlook outlook_ego;
        Outlook outlook_agent;
        Outlook outlook_steering;
        double steer_size = 5;
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
    
        std::vector<RoutingLine> routing_lines(vehicle_num);
        for (size_t idx = 0; idx < vehicle_num; ++idx) {
            size_t line_num = 0;
            double start_s = center_lines[line_num].length();
            double min_diff = -1.0;
            for (size_t l = 0; l < center_lines.size(); ++l) {
                for (size_t i = 1; i < center_lines[l].size(); ++i) {
                    double last_diff = hypot(center_lines[l].x[i - 1] - initial_conditions[idx][0],
                                             center_lines[l].y[i - 1] - initial_conditions[idx][1]);
                    double cur_diff = hypot(center_lines[l].x[i] - initial_conditions[idx][0],
                                            center_lines[l].y[i] - initial_conditions[idx][1]);
                    if (cur_diff > last_diff) {
                        if (min_diff < 0 || last_diff < min_diff) {
                            min_diff = last_diff;
                            line_num = l;
                            start_s = center_lines[l].longitude[i - 1];
                        }
                        break;
                    }
                }
            }
            SPDLOG_DEBUG("idx: {}, line_num: {}, start_s: {}", idx, line_num, start_s);
            // 定位初始车道
            for (double t = 0.0; t < max_simulation_time + 10; t += delta_t) {
                double cur_s = 0.;
                Eigen::Vector3d pos;
                // The current laneline does not have the attribute of driving direction,
                // and it is simply deduced by the yaw in the initial condition.
                if (initial_conditions[idx][3] <= M_PI_2) {
                    cur_s = start_s + t * initial_conditions[idx][2];
                    cur_s = std::min(cur_s, center_lines[line_num].longitude.back());
                    pos = center_lines[line_num].calc_position(cur_s);
                } else {
                    cur_s = start_s - t * initial_conditions[idx][2];
                    cur_s = std::max(cur_s, center_lines[line_num].longitude.front());
                    pos = center_lines[line_num].calc_position(cur_s);
                    pos.z() = fmod(pos.z() + M_PI, 2 * M_PI);
                }
    
                // randomly add noise to other cars
                // TODO: the current planning results are very sensitive to noise and
                //       initial conditions, which need to be optimized.
                if (idx == 0 || Random::uniform(0.0, 1.0) < 0.5) {
                    routing_lines[idx].x.push_back(pos.x());
                    routing_lines[idx].y.push_back(pos.y());
                    routing_lines[idx].yaw.push_back(pos.z());
                } else {
                    routing_lines[idx].x.push_back(pos.x() + Random::normal(0.0, 0.02));
                    routing_lines[idx].y.push_back(pos.y() + Random::normal(0.0, 0.02));
                    routing_lines[idx].yaw.push_back(pos.z());
                }
            }
        }
        std::vector<RoutingLine> obs_prediction(routing_lines.begin() + 1, routing_lines.end()); // 预测结果
    
        Eigen::Vector4d ego_state = {initial_conditions[0][0], initial_conditions[0][1],
                                     initial_conditions[0][2], initial_conditions[0][3]};
        CILQRSolver ilqr_solver = CILQRSolver(config);
    
        for (double t = 0.; t < max_simulation_time; t += delta_t) {
            size_t index = t / delta_t;
            plt::cla();
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
    
            auto [new_u, new_x] =
                ilqr_solver.solve(ego_state, center_lines[0], target_velocity,
                                  get_sub_routing_lines(obs_prediction, index), road_borders);
            ego_state = new_x.row(1).transpose();
    
            Eigen::MatrixX4d boarder = utils::get_boundary(new_x, VEHICLE_WIDTH * 0.7);
            std::vector<std::vector<double>> closed_curve = utils::get_closed_curve(boarder);
            plt::fill(closed_curve[0], closed_curve[1], {{"color", "cyan"}, {"alpha", "0.7"}});
    
            utils::show_vehicle(outlook_ego, ego_state, vehicle_para, reference_point, wheelbase);
            for (size_t idx = 1; idx < vehicle_num; ++idx) {
                utils::show_vehicle(outlook_agent, routing_lines[idx][index], vehicle_para,
                                    reference_point, wheelbase);
            }
    
            if (show_reference_line) {
                plt::plot(center_lines[0].x, center_lines[0].y, "-r");
            }
    
            // defualt figure x-y limit
            double visual_x_min = ego_state.x() - 10;
            double visual_y_min = ego_state.y() - 5;
            double visual_x_max = ego_state.x() + 30;
            double visual_y_max = ego_state.y() + 15;
            if (hypot(visual_x_limit[0], visual_x_limit[1]) > 1e-3) {
                visual_x_min = visual_x_limit[0];
                visual_x_max = visual_x_limit[1];
            }
            if (hypot(visual_y_limit[0], visual_y_limit[1]) > 1e-3) {
                visual_y_min = visual_y_limit[0];
                visual_y_max = visual_y_limit[1];
            }
    
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
            plt::text(text_left + 10, text_top, fmt::format("acc = {:.2f}", new_u.row(0)[0]),
                      {{"color", "black"}});
            plt::text(text_left + 10, text_top - 1.5, fmt::format("steer = {:.2f}", new_u.row(0)[1]),
                      {{"color", "black"}});
    
            plt::xlim(visual_x_min, visual_x_max);
            plt::ylim(visual_y_min, visual_y_max);
            plt::pause(delta_t);
        }
    
        config->destroy_instance();
        plt::show();
    
        return 0;
    }
};

namespace py = pybind11;

PYBIND11_MODULE(motion_planning,m)
{
  m.doc() = "pybind11 motion_planning plugin";

  py::class_<api_solver>(m, "api_solver")
  .def(py::init())
  .def("demo", &api_solver::demo);
}