#include <memory>
#include <algorithm>
#include <string>
#include <cstdint>
#include <cstddef>
#include "torch_server_ops.hpp"
#include <istream>
#include <streambuf>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>


// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


/*
A preallocated chunk of memory, required to convert a char array to an istream.
*/
class membuf : public std::basic_streambuf<char> {
public:
  membuf(const char *p, size_t l) {
    this->setg((char *)p, (char *)p, (char *)p + l);
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which = std::ios_base::in) override {
    if (dir == std::ios_base::cur)
      gbump(off);
    else if (dir == std::ios_base::end)
      setg(eback(), egptr() + off, egptr());
    else if (dir == std::ios_base::beg)
      setg(eback(), eback() + off, egptr());
    return gptr() - eback();
  }

  pos_type seekpos(pos_type sp, std::ios_base::openmode which) override {
    return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
  }
  };

/**
TODO
*/
class memstream : public std::istream {
public:
  memstream(const char *p, size_t l) : std::istream(&_buffer), _buffer(p, l) {
    rdbuf(&_buffer);
  }

private:
  membuf _buffer;
};

struct TorchTensor {
  torch::Tensor data;
};
struct TorchScriptModule {
  torch::jit::script::Module data;
};
struct TorchInput {
  std::vector<torch::jit::IValue> data;
};

struct StateDict {
  c10::Dict<std::string, torch::Tensor> data;
};

TorchRobotState::TorchRobotState(int num_dofs) {
  num_dofs_ = num_dofs;

  // Create initial state dictionary
  rs_timestamp_ = new TorchTensor{torch::zeros(2).to(torch::kInt32)};
  rs_joint_positions_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_joint_velocities_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_motor_torques_measured_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_motor_torques_external_ = new TorchTensor{torch::zeros(num_dofs)};

  state_dict_ = new StateDict{c10::Dict<std::string, torch::Tensor>()};

  state_dict_->data.insert("timestamp", rs_timestamp_->data);
  state_dict_->data.insert("joint_positions", rs_joint_positions_->data);
  state_dict_->data.insert("joint_velocities", rs_joint_velocities_->data);
  state_dict_->data.insert("motor_torques_measured",
                           rs_motor_torques_measured_->data);
  state_dict_->data.insert("motor_torques_external",
                           rs_motor_torques_external_->data);

  input_ = new TorchInput{std::vector<torch::jit::IValue>()};
  input_->data.push_back(state_dict_->data);
}

TorchRobotState::~TorchRobotState() {
  delete rs_timestamp_;
  delete rs_joint_positions_;
  delete rs_joint_velocities_;
  delete rs_motor_torques_measured_;
  delete rs_motor_torques_external_;
  delete state_dict_;
  delete input_;
}

void TorchRobotState::update_state(int timestamp_s, int timestamp_ns,
                                   std::vector<float> joint_positions,
                                   std::vector<float> joint_velocities,
                                   std::vector<float> motor_torques_measured,
                                   std::vector<float> motor_torques_external) {
  rs_timestamp_->data[0] = timestamp_s;
  rs_timestamp_->data[1] = timestamp_ns;
  for (int i = 0; i < joint_positions.size(); i++) {
    rs_joint_positions_->data[i] = joint_positions[i];
    rs_joint_velocities_->data[i] = joint_velocities[i];
    rs_motor_torques_measured_->data[i] = motor_torques_measured[i];
    rs_motor_torques_external_->data[i] = motor_torques_external[i];
  }
}

TorchScriptedController::TorchScriptedController(
    char *data, size_t size, TorchRobotState &init_robot_state) {
  memstream stream(data, size);
  module_ = new TorchScriptModule{torch::jit::load(stream)};

  param_dict_input_ = new TorchInput{std::vector<torch::jit::IValue>()};
  empty_input_ = new TorchInput{std::vector<torch::jit::IValue>()};

  // Warm up controller (TorchScript models take time to compile during first 2
  // queries)
  this->warmup_controller(WARM_UP_ITERS, init_robot_state);
}

TorchScriptedController::~TorchScriptedController() {
  delete module_;
  delete param_dict_input_;
  delete empty_input_;
}

std::vector<float> TorchScriptedController::forward(TorchRobotState &input) {
  torch::NoGradGuard no_grad;
  // Step controller & generate torque command response
  c10::Dict<torch::jit::IValue, torch::jit::IValue> controller_state_dict =
      module_->data.forward(input.input_->data).toGenericDict();

  torch::jit::IValue key = torch::jit::IValue("joint_torques");
  torch::Tensor desired_torque = controller_state_dict.at(key).toTensor();

  std::vector<float> result;
  for (int i = 0; i < input.num_dofs_; i++) {
    result.push_back(desired_torque[i].item<float>());
  }
  return result;
}

void TorchScriptedController::warmup_controller(
    int warmup_iters, TorchRobotState &init_robot_state) {
  // Backup
  auto tmp_module_data = module_->data.deepcopy();

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    this->forward(init_robot_state);
  }

  // Reload
  module_->data = tmp_module_data;
}

bool TorchScriptedController::is_terminated() {
  return module_->data.get_method("is_terminated")(empty_input_->data).toBool();
}

void TorchScriptedController::reset() {
  module_->data.get_method("reset")(empty_input_->data);
}

bool TorchScriptedController::param_dict_load(char *data, size_t size) {
  memstream model_stream(data, size);

  torch::jit::script::Module param_dict_container;
  try {
    param_dict_container = torch::jit::load(model_stream);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the param container:\n";
    std::cerr << e.msg() << std::endl;
    return false;
  }

  // Create controller update input dict
  param_dict_input_->data.clear();
  param_dict_input_->data.push_back(
      param_dict_container.forward(empty_input_->data));

  return true;
}

void TorchScriptedController::param_dict_update_module() {
  module_->data.get_method("update")(param_dict_input_->data);
}

// --- ADDED BY REPAIR SCRIPT V4 (THE FINAL LIBRARY) ---

// 1. Quaternion Multiplication
torch::Tensor quaternion_multiply(torch::Tensor q1, torch::Tensor q2) {
    auto w1 = q1.select(-1, 0); auto x1 = q1.select(-1, 1);
    auto y1 = q1.select(-1, 2); auto z1 = q1.select(-1, 3);
    auto w2 = q2.select(-1, 0); auto x2 = q2.select(-1, 1);
    auto y2 = q2.select(-1, 2); auto z2 = q2.select(-1, 3);
    auto w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    auto x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    auto y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    auto z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    return torch::stack({w, x, y, z}, -1);
}

// 2. Quaternion to Matrix
torch::Tensor quat2matrix(torch::Tensor q) {
    auto w = q.select(-1, 0); auto x = q.select(-1, 1);
    auto y = q.select(-1, 2); auto z = q.select(-1, 3);
    auto x2 = x * x; auto y2 = y * y; auto z2 = z * z;
    auto xy = x * y; auto xz = x * z; auto yz = y * z;
    auto wx = w * x; auto wy = w * y; auto wz = w * z;
    auto r00 = 1 - 2 * (y2 + z2); auto r01 = 2 * (xy - wz); auto r02 = 2 * (xz + wy);
    auto r10 = 2 * (xy + wz); auto r11 = 1 - 2 * (x2 + z2); auto r12 = 2 * (yz - wx);
    auto r20 = 2 * (xz - wy); auto r21 = 2 * (yz + wx); auto r22 = 1 - 2 * (x2 + y2);
    auto row0 = torch::stack({r00, r01, r02}, -1);
    auto row1 = torch::stack({r10, r11, r12}, -1);
    auto row2 = torch::stack({r20, r21, r22}, -1);
    return torch::stack({row0, row1, row2}, -2);
}

// 3. Quaternion to Rotation Vector
torch::Tensor quat2rotvec(torch::Tensor q) {
    auto w = q.select(-1, 0);
    auto v = q.slice(-1, 1, 4);
    auto norm_v = torch::norm(v, 2, -1);
    auto angle = 2.0 * torch::atan2(norm_v, w);
    auto scale = torch::where(norm_v > 1e-6, angle / norm_v, torch::zeros_like(angle));
    return v * scale.unsqueeze(-1);
}

// 4. Rotation Vector to Quaternion
torch::Tensor rotvec2quat(torch::Tensor rotvec) {
    auto angle = torch::norm(rotvec, 2, -1);
    auto scale = torch::where(angle > 1e-6, torch::sin(angle / 2.0) / angle, torch::zeros_like(angle));
    auto w = torch::cos(angle / 2.0);
    auto v = rotvec * scale.unsqueeze(-1);
    return torch::cat({w.unsqueeze(-1), v}, -1);
}

// 5. Quaternion to Axis
torch::Tensor quat2axis(torch::Tensor q) {
    auto v = q.slice(-1, 1, 4);
    auto norm_v = torch::norm(v, 2, -1, true);
    auto fallback = torch::tensor({0.0, 0.0, 1.0}, q.options()).expand_as(v);
    return torch::where(norm_v > 1e-6, v / norm_v, fallback);
}

// 6. Quaternion to Angle
torch::Tensor quat2angle(torch::Tensor q) {
    auto w = q.select(-1, 0);
    auto v = q.slice(-1, 1, 4);
    auto norm_v = torch::norm(v, 2, -1);
    return 2.0 * torch::atan2(norm_v, w);
}

// 7. Skew Symmetric Matrix
torch::Tensor skew_symmetric(torch::Tensor v) {
    auto x = v.select(-1, 0);
    auto y = v.select(-1, 1);
    auto z = v.select(-1, 2);
    auto zeros = torch::zeros_like(x);
    auto row0 = torch::stack({zeros, -z, y}, -1);
    auto row1 = torch::stack({z, zeros, -x}, -1);
    auto row2 = torch::stack({-y, x, zeros}, -1);
    return torch::stack({row0, row1, row2}, -2);
}

// 8. Invert Quaternion (The one you crashed on)
torch::Tensor invert_quaternion(torch::Tensor q) {
    // For unit quaternions, inverse is just conjugate [w, -x, -y, -z]
    auto w = q.select(-1, 0);
    auto v = q.slice(-1, 1, 4);
    return torch::cat({w.unsqueeze(-1), -v}, -1);
}

// 9. Rotate Vector (Likely next crash prevention)
torch::Tensor rotate_vector(torch::Tensor q, torch::Tensor v) {
    // v_rot = q * v * q_inv
    // Convert vector v to pure quaternion [0, v]
    auto zeros = torch::zeros_like(v.select(-1, 0)).unsqueeze(-1);
    auto q_v = torch::cat({zeros, v}, -1);
    
    auto q_inv = invert_quaternion(q);
    auto temp = quaternion_multiply(q, q_v);
    auto result = quaternion_multiply(temp, q_inv);
    
    // Return imaginary part
    return result.slice(-1, 1, 4);
}

// REGISTRATION
TORCH_LIBRARY(torch_server_ops, m) {
    m.def("quaternion_multiply", &quaternion_multiply);
    m.def("quat2matrix", &quat2matrix);
    m.def("quat2rotvec", &quat2rotvec);
    m.def("rotvec2quat", &rotvec2quat);
    m.def("quat2axis", &quat2axis);
    m.def("quat2angle", &quat2angle);
    m.def("skew_symmetric", &skew_symmetric);
    m.def("invert_quaternion", &invert_quaternion);
    m.def("rotate_vector", &rotate_vector);
}