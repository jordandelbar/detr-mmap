# Bridge-RT
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Build](https://github.com/jordandelbar/bridge-rt/actions/workflows/ci.yaml/badge.svg)](https://github.com/jordandelbar/bridge-rt/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/jordandelbar/bridge-rt/branch/main/graph/badge.svg?token=GFI0VJOZ9G)](https://codecov.io/gh/jordandelbar/bridge-rt)

A RT-DERT implementation with Rust ort, flatbuffers and k3d

## 📝 Overview

This project implements a real-time object detection on edge using a RT-DETR v2 model. It is made of 4 services that works together:

- capture: captures video frame from the camera using the [nokhwa] crate and v4l
- inference: runs an [ort] inference service
- controller: state machine to track if a person is present in the detection and communicate with the capture service to change its mode to standby or alarmed.
- gateway: provide a websocket endpoint to allow the user to see the frame with the detections

## Tech Stack

- nokhwa for capture
- Ort for running RT-DETR model inference
- MQTT
- HTML + JavaScript for real-time visualization

## Architecture

## Technical decisions
- "Why mmap instead of gRPC/HTTP?"
- "Why separate processes instead of threads?"
- "Why FlatBuffers over Protobuf?"

## Installation

## ⚡ Running with CUDA

## Performance & Benchmarks
