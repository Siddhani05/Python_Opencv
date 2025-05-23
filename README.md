﻿# Python_Opencv
With the increasing use of online platforms for streaming content such as YouTube and OTT (Over-The-Top) services, there is a growing demand for touchless interaction methods. Traditional input devices like keyboards, remotes, or mice can be inconvenient or inaccessible in certain scenarios, especially during presentations, while cooking, or in smart home environments. This project addresses the problem of implementing an efficient and intuitive gesture recognition system that allows users to control media playback through simple hand gestures using a webcam. The objective is to ensure smooth, real-time control that enhances user comfort and accessibility.

Module wise description:
-Module 1: Hand Detection and Tracking
Utilizes the MediaPipe Hands solution to detect 21 hand landmarks.
Tracks one hand per frame.
Processes real-time video input to continuously update hand position.
-Module 2: Gesture Recognition Engine
Analyzes the positions of specific hand landmarks.
Determines the up/down status of each finger using relative coordinates.
Matches finger combinations to specific gestures.
Recognized gestures: volume up, volume down, play/pause, forward, backward, next video, mute/unmute, fullscreen.
-Module 3: Gesture-to-Action Mapping
Uses PyAutoGUI to simulate corresponding keyboard presses.
Contains platform-specific mappings:
YouTube: Uses keys like K (pause), M (mute), J (back), L (forward).
OTT: Uses keys like Space (pause), Right Arrow (forward), F11 (fullscreen).
Introduces a cooldown timer to prevent repeated rapid actions.
-Module 4: Face Detection (Pause on Absence)
Uses MediaPipe Face Detection to ensure user presence.
Automatically pauses media if no face is detected in front of the webcam.
-Module 5: Graphical User Interface (GUI)
Built using Python's Tkinter library.
Presents a dark-themed aesthetic.
Allows users to select YouTube or OTT gesture control mode.
Displays instructions and project branding.

