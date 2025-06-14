import { Instance } from "../instance";

class VideoService {
  static createWebSocketConnection(videoSessionId) {
    if (!videoSessionId) {
      throw new Error("Video session ID is required for WebSocket connection");
    }
    
    const baseUrl = Instance.defaults.baseURL;
    const wsUrl = baseUrl.replace(/^http/, 'ws') + `/ws/video_recognition/${videoSessionId}`;
    console.log("Creating WebSocket connection to:", wsUrl);
    
    return new WebSocket(wsUrl);
  }

  static async startVideoProcessing(userId, videoUrl, params = {}) {
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
      
      const session = response.data;
      console.log("Video session created:", session);
      
      const sessionId = session.id;
      
      if (!sessionId) {
        throw new Error("No session ID returned from server");
      }
      
      return {
        session,
        sessionId
      };
    } catch (error) {
      console.error("Error starting video processing:", error);
      throw error;
    }
  }

  static createWebSocketAndInitialize(sessionId, videoUrl, params = {}) {
    return new Promise((resolve, reject) => {
      const wsConnection = this.createWebSocketConnection(sessionId);
      
      const timeout = setTimeout(() => {
        wsConnection.close();
        reject(new Error("WebSocket connection timeout"));
      }, 10000);

      wsConnection.onopen = () => {
        clearTimeout(timeout);
        console.log("WebSocket connected successfully");
        
        const initMessage = {
          video_url: videoUrl,
          params: {
            max_duration: 60,
            tracker_type: "botsort",
            window_size_ratio: [0.7, 0.7],
            overlap_ratio: [0.1, 0.1],
            img_size: 640,
            conf: 0.1,
            iou: 0.1,
            nms_global: 0.1,
            classes: [0],
            include_annotated_frame: false,
            show_labels: false,
            ...params
          }
        };
        
        console.log("Sending init message:", initMessage);
        wsConnection.send(JSON.stringify(initMessage));
        resolve(wsConnection);
      };

      wsConnection.onerror = (error) => {
        clearTimeout(timeout);
        console.error("WebSocket connection error:", error);
        reject(error);
      };

      wsConnection.onclose = (event) => {
        clearTimeout(timeout);
        console.log("WebSocket closed with code:", event.code, "reason:", event.reason);
        if (event.code !== 1000) {
          reject(new Error(`WebSocket closed unexpectedly: ${event.code} - ${event.reason}`));
        }
      };
    });
  }

  static async getVideoSessionById(sessionId) {
    try {
      const response = await Instance.get(`/video-sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error(`Error getting video session ${sessionId}:`, error);
      throw error;
    }
  }

  static async getFrameAnnotations(videoSessionId) {
    try {
      const response = await Instance.get(
        `/frame-annotations/video-session/${videoSessionId}`
      );
      return response.data;
    } catch (error) {
      console.error(
        `Error getting frame annotations for session ${videoSessionId}:`,
        error
      );
      throw error;
    }
  }

  static async incrementUserVideos(userId) {
    try {
      const response = await Instance.patch(`/users/${userId}/videos/increment`);
      return response.data;
    } catch (error) {
      console.error(`Error incrementing video count for user ${userId}:`, error);
      throw error;
    }
  }
}

export default VideoService;