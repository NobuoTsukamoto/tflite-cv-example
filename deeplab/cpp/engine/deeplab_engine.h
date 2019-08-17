/**
 * Copyright (c) 2019 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#ifndef DEEPLAB_ENGINE_H_
#define DEEPLAB_ENGINE_H_

#include <memory>
#include <vector>
#include <chrono>

#include "tensorflow/lite/interpreter.h"


extern std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context,
    const unsigned int num_of_threads);

extern std::vector<float> RunInference(const std::vector<uint8_t>& input_data,
                                tflite::Interpreter* interpreter,
                                std::chrono::duration<double, std::milli>& time_span);

#endif /* DEEPLAB_ENGINE_H_ */