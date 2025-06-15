import { Instance } from "../instance";

class VideoServiceError extends Error {
  constructor(message, code = null, originalError = null) {
    super(message);
    this.name = 'VideoServiceError';
    this.code = code;
    this.originalError = originalError;
  }
}

class VideoService {
  static createWebSocketConnection(videoSessionId) {
    if (!videoSessionId) {
      throw new VideoServiceError("Video session ID is required for WebSocket connection", "MISSING_SESSION_ID");
    }
    
    try {
      const baseUrl = Instance.defaults?.baseURL;
      if (!baseUrl) {
        throw new VideoServiceError("Base URL not configured", "MISSING_BASE_URL");
      }
      
      const wsUrl = baseUrl.replace(/^http/, 'ws') + `/ws/video_recognition/${videoSessionId}`;
      console.log("Creating WebSocket connection to:", wsUrl);
      
      return new WebSocket(wsUrl);
    } catch (error) {
      throw new VideoServiceError("Failed to create WebSocket connection", "WEBSOCKET_CREATION_FAILED", error);
    }
  }

  static async startVideoProcessing(userId, videoUrl, params = {}) {
    if (!userId || !videoUrl) {
      throw new VideoServiceError("User ID and video URL are required", "MISSING_PARAMETERS");
    }

    try {
      const formData = new URLSearchParams();
      formData.append("user_id", userId.toString());
      formData.append("video_url", videoUrl);

      console.log("Creating video session with data:", { userId, videoUrl });

      const response = await Instance.post("/video-sessions/", formData, {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      });
      
      if (!response?.data) {
        throw new VideoServiceError("Invalid response from server", "INVALID_RESPONSE");
      }
      
      const session = response.data;
      console.log("Video session created:", session);
      
      const sessionId = session.id;
      
      if (!sessionId) {
        throw new VideoServiceError("No session ID returned from server", "MISSING_SESSION_ID");
      }
      
      return { session, sessionId };

    } catch (error) {
      if (error instanceof VideoServiceError) {
        throw error;
      }
      
      if (error.code === 'ERR_NETWORK') {
        throw new VideoServiceError("Network error: Unable to connect to server", "NETWORK_ERROR", error);
      }
      
      if (error.response?.status) {
        throw new VideoServiceError(
          `Server error: ${error.response.status} - ${error.response.data?.message || 'Unknown error'}`,
          "SERVER_ERROR",
          error
        );
      }
      
      throw new VideoServiceError("Unexpected error starting video processing", "UNKNOWN_ERROR", error);
    }
  }

  static createWebSocketAndInitialize(sessionId, videoUrl, params = {}) {
    return new Promise((resolve, reject) => {
      let wsConnection = null;
      let isResolved = false;

      const timeout = setTimeout(() => {
        if (!isResolved && wsConnection) {
          wsConnection.close();
          reject(new VideoServiceError("WebSocket connection timeout", "WEBSOCKET_TIMEOUT"));
        }
      }, 10000);

      const cleanup = () => {
        if (timeout) {
          clearTimeout(timeout);
        }
        isResolved = true;
      };

      try {
        wsConnection = this.createWebSocketConnection(sessionId);
      } catch (error) {
        cleanup();
        reject(error);
        return;
      }
      
      wsConnection.onopen = () => {
        if (isResolved) return;
        
        cleanup();
        console.log("WebSocket connected successfully");
        
        try {
          const initMessage = {
            video_url: videoUrl,
            params: {
              max_duration: 60,
              tracker_type: "botsort",
              window_size_ratio: [1.0, 1.0],
              overlap_ratio: [0.0, 0.0],
              img_size: 640,
              conf: 0.5,
              iou: 0.1,
              nms_global: 0.1,
              classes: [0],
              include_annotated_frame: false,
              show_labels: false,
              tracker_detection_source: "local",
              model_path: "/home/student/projects/FrameReader/docs/last.engine",
              history_length: 24,
              triton_ws_url: null,
              triton_stream_url: null,
              triton_batch_url: null,
              donut_triton_stream_url: null,
              donut_triton_ws_url: null,
              donut_detection_source: "main",
              donut_model_name: "donut",
              triton_model_name: "yolo",
              ...params
            }
          };
          
          console.log("Sending init message:", initMessage);
          wsConnection.send(JSON.stringify(initMessage));
          resolve(wsConnection);
        } catch (error) {
          reject(new VideoServiceError("Failed to send init message", "INIT_MESSAGE_FAILED", error));
        }
      };

      wsConnection.onerror = (error) => {
        if (isResolved) return;
        
        cleanup();
        console.error("WebSocket connection error:", error);
        reject(new VideoServiceError("WebSocket connection error", "WEBSOCKET_ERROR", error));
      };

      wsConnection.onclose = (event) => {
        if (isResolved) return;
        
        cleanup();
        console.log("WebSocket closed with code:", event.code, "reason:", event.reason);
        
        if (event.code !== 1000) {
          reject(new VideoServiceError(
            `WebSocket closed unexpectedly: ${event.code} - ${event.reason}`,
            "WEBSOCKET_CLOSED",
            event
          ));
        }
      };
    });
  }

  static async getVideoSessionById(sessionId) {
    if (!sessionId) {
      throw new VideoServiceError("Session ID is required", "MISSING_SESSION_ID");
    }

    try {
      const response = await Instance.get(`/video-sessions/${sessionId}`);
      
      if (!response?.data) {
        throw new VideoServiceError("Invalid response from server", "INVALID_RESPONSE");
      }
      
      return response.data;
    } catch (error) {
      if (error instanceof VideoServiceError) {
        throw error;
      }
      
      throw new VideoServiceError(
        `Error getting video session ${sessionId}`,
        "GET_SESSION_FAILED",
        error
      );
    }
  }

  static async getFrameAnnotations(videoSessionId) {
    if (!videoSessionId) {
      throw new VideoServiceError("Video session ID is required", "MISSING_SESSION_ID");
    }

    try {
      const response = await Instance.get(`/frame-annotations/video-session/${videoSessionId}`);
      
      if (!response?.data) {
        throw new VideoServiceError("Invalid response from server", "INVALID_RESPONSE");
      }
      
      return response.data;
    } catch (error) {
      if (error instanceof VideoServiceError) {
        throw error;
      }
      
      throw new VideoServiceError(
        `Error getting frame annotations for session ${videoSessionId}`,
        "GET_ANNOTATIONS_FAILED",
        error
      );
    }
  }

  static async incrementUserVideos(userId) {
    if (!userId) {
      throw new VideoServiceError("User ID is required", "MISSING_USER_ID");
    }

    try {
      const response = await Instance.patch(`/users/${userId}/videos/increment`);
      
      if (!response?.data) {
        throw new VideoServiceError("Invalid response from server", "INVALID_RESPONSE");
      }
      
      return response.data;
    } catch (error) {
      if (error instanceof VideoServiceError) {
        throw error;
      }
      
      throw new VideoServiceError(
        `Error incrementing video count for user ${userId}`,
        "INCREMENT_VIDEOS_FAILED",
        error
      );
    }
  }
}

export default VideoService;
export { VideoServiceError };