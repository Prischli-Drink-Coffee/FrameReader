import React, { useRef, useEffect, useState, useCallback } from "react";
import { Box, Text, Spinner, Flex } from "@chakra-ui/react";
import { Instance } from "../API/instance";

const VideoPlayerWithAnnotations = ({ 
  videoUrl, 
  videoSessionId, 
  onProcessingComplete, 
  onProcessingError 
}) => {
  const playerContainerRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const rutubePlayerRef = useRef(null);
  
  const [statusMessage, setStatusMessage] = useState("Connecting to processing server...");
  const [currentAnnotations, setCurrentAnnotations] = useState([]);
  const [recognizedText, setRecognizedText] = useState("");
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const [playerReady, setPlayerReady] = useState(false);
  const [annotationsBuffer, setAnnotationsBuffer] = useState([]);
  const [currentFrameTime, setCurrentFrameTime] = useState(0);

  const FPS = 5;

  const extractVideoId = useCallback((url) => {
    const match = url.match(/rutube\.ru\/video\/([a-f0-9]{32})/);
    return match ? match[1] : null;
  }, []);

  const drawAnnotations = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !playerReady) return;

    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);

    currentAnnotations.forEach(annotation => {
      if (annotation.box) {
        const [x1, y1, x2, y2] = annotation.box;
        const width = x2 - x1;
        const height = y2 - y1;

        context.strokeStyle = annotation.color || "red";
        context.lineWidth = 2;
        context.strokeRect(x1, y1, width, height);

        if (annotation.recognized_text) {
          context.fillStyle = annotation.color || "red";
          context.font = "16px Arial";
          context.fillText(annotation.recognized_text, x1, y1 > 10 ? y1 - 5 : y1 + 15);
        }
      }
    });
  }, [currentAnnotations, playerReady]);

  const initializeRutubePlayer = useCallback(() => {
    const videoId = extractVideoId(videoUrl);
    if (!videoId || !window.Rutube) {
      console.error("Video ID not found or Rutube library not loaded");
      onProcessingError("Failed to initialize video player");
      return;
    }

    try {
      const rt = new window.Rutube();
      rutubePlayerRef.current = rt;

      rt.Player('rutube-player', {
        width: 840,
        height: 473,
        videoId: videoId,
        events: {
          onReady: 'onPlayerReady',
          onStateChange: 'onPlayerStateChange'
        }
      });

      window.onPlayerReady = (event) => {
        console.log('Rutube player ready:', event);
        setPlayerReady(true);
        setStatusMessage("Video player ready. Waiting for annotations...");
        setVideoDimensions({ width: 840, height: 473 });
        
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = 840;
          canvas.height = 473;
        }
      };

      window.onPlayerStateChange = (event) => {
        console.log('Player state change:', event);
        if (rutubePlayerRef.current) {
          const currentTime = rutubePlayerRef.current.currentDuration();
          if (currentTime !== undefined) {
            setCurrentFrameTime(currentTime);
          }
        }
      };

    } catch (error) {
      console.error("Error initializing Rutube player:", error);
      onProcessingError("Failed to initialize video player");
    }
  }, [videoUrl, extractVideoId, onProcessingError]);

  useEffect(() => {
    if (!videoSessionId) return;

    const baseUrl = Instance.defaults?.baseURL || 'http://localhost:8010/server';
    const wsUrl = `${baseUrl.replace(/^http/, 'ws')}/ws/video_recognition/${videoSessionId}`;
    
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setStatusMessage("WebSocket connected. Sending video URL...");
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ 
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
          }
        }));
      }
    };

    wsRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log("WS Message:", message);

        if (message.status) {
          setStatusMessage(message.message || "Processing...");
          if (message.status === "error") {
            onProcessingError(message.message);
            return;
          }
          if (message.message && message.message.includes("completed")) {
            setStatusMessage("Video processing completed.");
            onProcessingComplete();
            return;
          }
        } else if (message.frame_number !== undefined) {
          setAnnotationsBuffer(prev => [...prev, {
            timestamp: message.timestamp || 0,
            tracked_objects: message.tracked_objects || [],
            frame_number: message.frame_number
          }]);
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    wsRef.current.onclose = (event) => {
      console.log("WebSocket closed:", event);
      if (!event.wasClean && event.code !== 1000) {
        onProcessingError("WebSocket connection closed unexpectedly.");
      }
    };

    wsRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      onProcessingError("WebSocket error during processing.");
    };

    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [videoSessionId, videoUrl, onProcessingComplete, onProcessingError]);

  useEffect(() => {
    if (window.Rutube) {
      initializeRutubePlayer();
    } else {
      const checkRutube = setInterval(() => {
        if (window.Rutube) {
          clearInterval(checkRutube);
          initializeRutubePlayer();
        }
      }, 100);

      setTimeout(() => {
        clearInterval(checkRutube);
        if (!window.Rutube) {
          onProcessingError("Rutube player library failed to load");
        }
      }, 5000);
    }

    return () => {
      if (window.onPlayerReady) {
        delete window.onPlayerReady;
      }
      if (window.onPlayerStateChange) {
        delete window.onPlayerStateChange;
      }
    };
  }, [initializeRutubePlayer, onProcessingError]);

  useEffect(() => {
    if (!playerReady || annotationsBuffer.length === 0) return;

    const processBuffer = () => {
      const annotationsForCurrentTime = annotationsBuffer.filter(
        (ann) => Math.abs(ann.timestamp - currentFrameTime) < (1 / FPS)
      );

      if (annotationsForCurrentTime.length > 0) {
        const latestAnnotation = annotationsForCurrentTime.reduce((prev, current) =>
          (prev.timestamp > current.timestamp) ? prev : current
        );
        
        setCurrentAnnotations(latestAnnotation.tracked_objects || []);
        setRecognizedText(
          latestAnnotation.tracked_objects
            ?.map(obj => obj.recognized_text)
            .filter(Boolean)
            .join(" ") || ""
        );
      }

      drawAnnotations();
    };

    const intervalId = setInterval(processBuffer, 1000 / FPS);

    return () => {
      clearInterval(intervalId);
    };
  }, [playerReady, annotationsBuffer, currentFrameTime, drawAnnotations]);

  useEffect(() => {
    let syncInterval;
    
    if (playerReady && rutubePlayerRef.current) {
      syncInterval = setInterval(() => {
        try {
          const currentTime = rutubePlayerRef.current.currentDuration();
          if (currentTime !== undefined && currentTime !== currentFrameTime) {
            setCurrentFrameTime(currentTime);
          }
        } catch (error) {
          console.warn("Error getting player time:", error);
        }
      }, 200);
    }

    return () => {
      if (syncInterval) {
        clearInterval(syncInterval);
      }
    };
  }, [playerReady, currentFrameTime]);

  return (
    <Box position="relative" width="100%" maxW="840px" mx="auto" mt={4}>
      {!playerReady && (
        <Flex
          position="absolute"
          top="0"
          left="0"
          right="0"
          bottom="0"
          bg="black"
          color="white"
          align="center"
          justify="center"
          flexDirection="column"
          zIndex="1"
        >
          <Spinner size="xl" mb={4} />
          <Text>{statusMessage}</Text>
        </Flex>
      )}
      
      <Box 
        ref={playerContainerRef}
        position="relative"
        width="840px"
        height="473px"
        display={playerReady ? "block" : "none"}
      >
        <div id="rutube-player" style={{ width: '100%', height: '100%' }} />
        
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            zIndex: 10
          }}
        />
      </Box>
      
      {recognizedText && (
        <Box
          position="absolute"
          bottom="0"
          left="0"
          right="0"
          bg="rgba(0,0,0,0.7)"
          color="white"
          p={2}
          textAlign="center"
          fontSize="lg"
          zIndex="20"
        >
          <Text>{recognizedText}</Text>
        </Box>
      )}
    </Box>
  );
};

export default VideoPlayerWithAnnotations;