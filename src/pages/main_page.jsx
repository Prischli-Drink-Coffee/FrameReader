import { VStack, Box, Alert, AlertIcon, AlertTitle, AlertDescription, CloseButton } from "@chakra-ui/react";
import useWindowDimensions from "../hooks/window_dimensions";
import ContentSection from "../components/main_content";
import VideoPlayerWithAnnotations from "../components/video_player";
import React, { useState, useEffect, useCallback, useRef } from "react";
import SessionService from "../API/services/session_service";
import VideoService from "../API/services/video_service";

const MainPage = () => {
  const { height } = useWindowDimensions();
  const [userId, setUserId] = useState(null);
  const [videoSessionId, setVideoSessionId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState("idle");
  const [errorMessage, setErrorMessage] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [frameData, setFrameData] = useState([]);
  const wsConnectionRef = useRef(null);

  useEffect(() => {
    const initSession = async () => {
      try {
        const sessionInfo = await SessionService.createOrGetSession();
        setUserId(sessionInfo.user_id);
        console.log("Session initialized:", sessionInfo);
      } catch (error) {
        console.error("Failed to initialize session:", error);
        setErrorMessage("Failed to initialize session. Please try again later.");
      }
    };
    initSession();
  }, []);

  const handleProcessVideo = useCallback(async (url) => {
    if (!userId) {
      setErrorMessage("User session not initialized. Please refresh the page.");
      return;
    }

    setVideoUrl(url);
    setProcessingStatus("processing");
    setErrorMessage(null);
    setVideoSessionId(null);
    setFrameData([]);

    try {
      const { session, sessionId } = await VideoService.startVideoProcessing(userId, url);
      
      setVideoSessionId(sessionId);

      const wsConnection = await VideoService.createWebSocketAndInitialize(sessionId, url);
      wsConnectionRef.current = wsConnection;

      wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.status === "error") {
            setProcessingStatus("failed");
            setErrorMessage(data.message);
            return;
          }

          if (data.status === "info") {
            console.log("Processing update:", data.message);
            if (data.message && data.message.includes("completed")) {
              setProcessingStatus("completed");
            }
            return;
          }

          if (data.frame_number !== undefined) {
            console.log("Frame data received:", data);
            setFrameData(prev => [...prev, data]);
          }
        } catch (parseError) {
          console.error("Error parsing WebSocket message:", parseError);
        }
      };

      wsConnection.onclose = (event) => {
        console.log("WebSocket connection closed", event.code, event.reason);
        wsConnectionRef.current = null;
      };

      wsConnection.onerror = (error) => {
        console.error("WebSocket error:", error);
        setProcessingStatus("failed");
        setErrorMessage("Connection error during processing");
        wsConnectionRef.current = null;
      };
      
      console.log("Video processing started:", session);
    } catch (error) {
      console.error("Failed to start video processing:", error);
      setProcessingStatus("failed");
      setErrorMessage(
        error.response?.data?.detail || error.message || "Failed to start video processing. Please check the URL and try again."
      );
    }
  }, [userId]);

  const handleProcessingComplete = useCallback(async () => {
    setProcessingStatus("completed");
    console.log("Video processing completed.");
    
    if (wsConnectionRef.current) {
      wsConnectionRef.current.close();
      wsConnectionRef.current = null;
    }
    
    if (userId) {
      try {
        await VideoService.incrementUserVideos(userId);
      } catch (error) {
        console.error("Failed to increment user video count:", error);
      }
    }
  }, [userId]);

  const handleProcessingError = useCallback((message) => {
    setProcessingStatus("failed");
    setErrorMessage(message);
    
    if (wsConnectionRef.current) {
      wsConnectionRef.current.close();
      wsConnectionRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (wsConnectionRef.current && wsConnectionRef.current.readyState === WebSocket.OPEN) {
        wsConnectionRef.current.close();
      }
    };
  }, []);

  return (
    <VStack
      minH="100vh"
      width="100%"
      align="center"
      justify="center"
      bg="#ffffff"
      padding={[4, 8, 16]}
      spacing={["16px", "20px", "30px"]}
      mt={["-40px", "-60px", "-80px"]}
      flexGrow={1}
    >
      <Box
        mt={["20px", "40px", "80px"]}
        width="100%"
        maxW="1200px"
        display="flex"
        flexDirection="column"
        alignItems="center"
        bg="#ffffff"
      >
        {errorMessage && (
          <Alert status="error" mb={4}>
            <AlertIcon />
            <AlertTitle mr={2}>Error!</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
            <CloseButton position="absolute" right="8px" top="8px" onClick={() => setErrorMessage(null)} />
          </Alert>
        )}

        <ContentSection onProcessVideo={handleProcessVideo} processingStatus={processingStatus} />

        {(processingStatus === "processing" || processingStatus === "completed") && 
         videoSessionId && 
         videoUrl && (
          <VideoPlayerWithAnnotations
            videoUrl={videoUrl || ""}
            videoSessionId={videoSessionId}
            frameData={frameData}
            processingStatus={processingStatus}
            onProcessingComplete={handleProcessingComplete}
            onProcessingError={handleProcessingError}
          />
        )}
      </Box>
    </VStack>
  );
};

export default MainPage;