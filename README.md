# Overview
### Motivation

Developing an open vocabulary
mobile manipulator (OVMM)
through LLM.
### Functionalities
Tayseer Should be able to help you with:
1) Sliding Objects
2) Getting Objects to you
3) Moving from one place to another

# Architecture
## The system consists of four main modules:
1) The Commender
2) The Virtual System Interconnect (VSI)
3) The Simulator/Real Robot (Isaac sim/Limo pro)
4) The ROS Packages for The Robot's Functionalities

## 1 - The Commander
#### Input: The command prompt, Feedback
#### Output: The commands the Robot must take in order
#### Responsibilities:
1) Receives natural language commands from the user.
2) Interprets the command using an LLM, informed by the robot’s available capabilities.
3) Decomposes the command into a structured, step-by-step plan.
4) Sends the plan to the Orchestrator for execution.
5) Monitors execution status and incorporates feedback to decide the next command.
## 2 - The Virtual System Interconnect (VSI)
For that we will use Siemens' Innexis VSI which is a tool developed by Siemens for digital twin application development 

 It represents our communication backbone to connect different clients
## 3- The Simulator/Real Robot (Isaac sim/Limo pro):
### Isaac Sim:
- Isaac Sim is a robotics simulation platform built on NVIDIA
Omniverse.

- It provides realistic 3D environments and sensor data for
testing robot perception and navigation.


- It enables seamless prototyping and evaluation of the robot‘s
functionalities in simulation before deploying in the real
world.
### Real Robot:
- #### The robot: Limo Pro (developed by AgileX)
- #### Attached arm: Mycobot280 M5

## 4- ROS Packages for the Robot's Functionalities:
#### Input: Command
#### Output: Feedback & Control to the robot
#### Responsibilities
##### 1) Perception: 
The robot is aware of his surroundings and target objects through the camera
##### 2) SLAM: 
The robot is capable of simultaneously localizing itself and mapping.
##### 3) Navigation: 
Gives the robot the ability to navigate from one point to another taking in consideration the obstacles avoidance and cost of the path
##### 4) Pick: 
Using the arm of the robot to pick the target object if applicable
##### 5) Place: 
Placing the target object correctly
##### 6) Sliding: 
Sliding an object with a certain amount of power to make it reach a certain place


