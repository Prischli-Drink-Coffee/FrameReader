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
  const [frameDataStream, setFrameDataStream] = useState(null);
  const [allFrameData, setAllFrameData] = useState([]);
  const wsConnectionRef = useRef(null);
  const frameCounterRef = useRef(0);

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

  const resetVideoProcessing = useCallback(() => {
    setVideoUrl("");
    setProcessingStatus("idle");
    setErrorMessage(null);
    setVideoSessionId(null);
    setFrameDataStream(null);
    setAllFrameData([]);
    frameCounterRef.current = 0;
    
    if (wsConnectionRef.current) {
      wsConnectionRef.current.close();
      wsConnectionRef.current = null;
    }
  }, []);

  const handleNewFrameData = useCallback((newFrameData) => {
    frameCounterRef.current += 1;
    
    console.log(`Frame data received #${frameCounterRef.current}:`, newFrameData);
    
    setFrameDataStream(newFrameData);
    setAllFrameData(prev => [...prev, newFrameData]);
  }, []);

  const handleProcessVideo = useCallback(async (url) => {
    if (!userId) {
      setErrorMessage("User session not initialized. Please refresh the page.");
      return;
    }

    resetVideoProcessing();
    
    setVideoUrl(url);
    setProcessingStatus("processing");

    try {
      const { session, sessionId } = await VideoService.startVideoProcessing(userId, url);
      setVideoSessionId(sessionId);

      const wsConnection = await VideoService.createWebSocketAndInitialize(sessionId, url);
      wsConnectionRef.current = wsConnection;

      wsConnection.onopen = () => {
        console.log("WebSocket connection established for session:", sessionId);
      };

      wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.status === "error") {
            setProcessingStatus("failed");
            setErrorMessage(data.message || "Processing failed");
            return;
          }

          if (data.status === "info") {
            console.log("Processing update:", data.message);
            if (data.message?.includes("completed")) {
              setProcessingStatus("completed");
            } else if (data.message?.includes("started")) {
              console.log("Video processing started successfully");
            }
            return;
          }

          if (data.frame_number !== undefined && data.timestamp !== undefined) {
            handleNewFrameData(data);
          }
        } catch (parseError) {
          console.error("Error parsing WebSocket message:", parseError);
        }
      };

      wsConnection.onclose = (event) => {
        console.log("WebSocket connection closed:", {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
        
        if (event.code !== 1000 && processingStatus === "processing") {
          setErrorMessage("Connection lost during processing");
          setProcessingStatus("failed");
        }
        
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
        error.response?.data?.detail || 
        error.message || 
        "Failed to start video processing. Please check the URL and try again."
      );
    }
  }, [userId, resetVideoProcessing, handleNewFrameData, processingStatus]);

  const handleProcessingComplete = useCallback(async () => {
    console.log(`Video processing completed. Total frames processed: ${frameCounterRef.current}`);
    setProcessingStatus("completed");
    
    if (wsConnectionRef.current) {
      wsConnectionRef.current.close();
      wsConnectionRef.current = null;
    }
    
    if (userId) {
      try {
        await VideoService.incrementUserVideos(userId);
        console.log("User video count incremented successfully");
      } catch (error) {
        console.error("Failed to increment user video count:", error);
      }
    }
  }, [userId]);

  const handleProcessingError = useCallback((message) => {
    console.error("Processing error:", message);
    setProcessingStatus("failed");
    setErrorMessage(message);
    
    if (wsConnectionRef.current) {
      wsConnectionRef.current.close();
      wsConnectionRef.current = null;
    }
  }, []);

  const closeErrorAlert = useCallback(() => {
    setErrorMessage(null);
  }, []);

  useEffect(() => {
    return () => {
      if (wsConnectionRef.current?.readyState === WebSocket.OPEN) {
        wsConnectionRef.current.close(1000, "Component unmounting");
      }
    };
  }, []);

  const shouldShowPlayer = Boolean(
    videoUrl && 
    videoSessionId && 
    (processingStatus === "processing" || processingStatus === "completed")
  );

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
          <Alert 
            status="error" 
            mb={4}
            borderRadius="12px"
            boxShadow="0 4px 12px rgba(245, 101, 101, 0.15)"
          >
            <AlertIcon />
            <Box flex="1">
              <AlertTitle mr={2}>Ошибка обработки!</AlertTitle>
              <AlertDescription>{errorMessage}</AlertDescription>
            </Box>
            <CloseButton 
              position="absolute" 
              right="8px" 
              top="8px" 
              onClick={closeErrorAlert}
            />
          </Alert>
        )}

        <ContentSection 
          onProcessVideo={handleProcessVideo} 
          processingStatus={processingStatus} 
        />

        {shouldShowPlayer && (
          <VideoPlayerWithAnnotations
            videoUrl={videoUrl}
            videoSessionId={videoSessionId}
            frameDataStream={frameDataStream}
            allFrameData={allFrameData}
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