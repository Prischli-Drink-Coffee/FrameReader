import React, { useRef, useEffect, useState, useCallback } from "react";
import { Box, Text, Spinner, Flex, Alert, AlertIcon } from "@chakra-ui/react";
import { keyframes } from "@emotion/react";

const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const scaleIn = keyframes`
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
`;

const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(75, 139, 252, 0.4);
  }
  70% {
    box-shadow: 0 0 0 15px rgba(75, 139, 252, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(75, 139, 252, 0);
  }
`;

const glowBorder = keyframes`
  0% {
    border-color: #4B8BFC;
    box-shadow: 0 0 20px rgba(75, 139, 252, 0.3);
  }
  50% {
    border-color: #667eea;
    box-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
  }
  100% {
    border-color: #4B8BFC;
    box-shadow: 0 0 20px rgba(75, 139, 252, 0.3);
  }
`;

const VideoPlayerWithAnnotations = ({ 
  videoUrl, 
  videoSessionId, 
  onProcessingComplete, 
  onProcessingError,
  frameDataStream,
  allFrameData,
  processingStatus: externalProcessingStatus
}) => {
  const canvasRef = useRef(null);
  const playerContainerRef = useRef(null);
  const rutubePlayerRef = useRef(null);
  const frameQueueRef = useRef([]);
  const currentFrameIndexRef = useRef(-1);
  const isVideoInitializedRef = useRef(false);
  const isProcessingNewFrameRef = useRef(false);
  
  const [statusMessage, setStatusMessage] = useState("Инициализация плеера...");
  const [currentAnnotations, setCurrentAnnotations] = useState([]);
  const [recognizedTexts, setRecognizedTexts] = useState([]);
  const [playerReady, setPlayerReady] = useState(false);
  const [error, setError] = useState(null);
  const [currentFrameData, setCurrentFrameData] = useState(null);
  const [waitingForNextFrame, setWaitingForNextFrame] = useState(false);
  const [uniqueTexts, setUniqueTexts] = useState(new Set());
  const [allFoundTexts, setAllFoundTexts] = useState([]);

  const extractVideoId = useCallback((url) => {
    if (!url || typeof url !== 'string') {
      throw new Error("Invalid video URL");
    }
    
    const match = url.match(/rutube\.ru\/video\/([a-f0-9]{32})/);
    if (!match) {
      throw new Error("Could not extract video ID from URL");
    }
    
    return match[1];
  }, []);

  const initializeRutubePlayer = useCallback(() => {
    if (typeof window === 'undefined') return;

    const Rutube = function () {
      const EMBEDED_API_URI = '//rutube.ru/play/embed/';
      const PREFFIX_PLAYER_ID = 'rt-';

      this.Player = function (selector, config) {
        if (!selector) {
          throw new Error('The Player element must be specified.');
        }

        this.selector = selector;
        this.config = config;
        this.duration = null;
        this.videoCurrentDuration = 0;

        this.renderOnPage();
      };

      this.renderOnPage = function () {
        const options = {
          id: PREFFIX_PLAYER_ID + this.selector,
          width: this.config.width || 720,
          height: this.config.height || 405,
          src: EMBEDED_API_URI + this.config.videoId + '?autoplay=0',
          frameBorder: 0,
          allow: 'autoplay',
          allowFullScreen: '',
          webkitallowfullscreen: '',
          mozallowfullscreen: '',
        };

        const element = document.createElement('iframe');

        for (let property in options) {
          element.setAttribute(property, options[property]);
        }

        const container = document.getElementById(this.selector);
        if (container) {
          container.innerHTML = '';
          container.appendChild(element);
        }
      };

      this.triggerEventObserver = function (env, args = null) {
        if (!this.config.events || !this.config.events[env]) return;
        return this.config.events[env](args);
      };

      this.setPlayerState = function (status) {
        const playerState = {
          PLAYING: 0,
          PAUSED: 0,
          STOPPED: 0,
          ENDED: 0,
        };

        for (let state in playerState) {
          if (state.toLowerCase() === status.toLowerCase()) {
            playerState[state] = 1;
            break;
          }
        }

        return { playerState };
      };

      this.currentDuration = function () {
        return this.videoCurrentDuration;
      };

      for (let [iterator, type] of Object.entries({
        play: 'play',
        pause: 'pause',
        stop: 'stop',
        seekTo: 'setCurrentTime',
        changeVideo: 'changeVideo',
        mute: 'mute',
        unMute: 'unMute',
        setVolume: 'setVolume',
      })) {
        this[iterator] = function (data = {}) {
          const iframe = document.getElementById(PREFFIX_PLAYER_ID + this.selector);
          if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage(
              JSON.stringify({
                type: 'player:' + type,
                data: data,
              }),
              '*'
            );
          }
        };
      }

      this.playerEvent = function (receivedMessage) {
        switch (receivedMessage.type) {
          case 'player:ready':
            this.triggerEventObserver('onReady', {
              videoId: receivedMessage.data.videoId,
              clientId: receivedMessage.data.clientId,
            });
            break;
          case 'player:changeState':
          case 'player:playComplete':
            this.triggerEventObserver(
              'onStateChange',
              this.setPlayerState(receivedMessage.data.state || 'ENDED')
            );
            break;
          case 'player:currentTime':
            this.videoCurrentDuration = receivedMessage.data.time;
            break;
        }
      };

      window.addEventListener(
        'message',
        function (event) {
          try {
            const receivedMessage = JSON.parse(event.data);
            this.playerEvent(receivedMessage);
          } catch (e) {
            console.warn('Failed to parse player message:', e);
          }
        }.bind(this),
        false
      );
    };

    window.Rutube = Rutube;

    try {
      const videoId = extractVideoId(videoUrl);
      const rt = new Rutube();
      
      const playerConfig = {
        width: 840,
        height: 473,
        videoId: videoId,
        events: {
          onReady: (event) => {
            console.log('Rutube Player готов:', event);
            setPlayerReady(true);
            setStatusMessage("Плеер готов! Ожидание аннотаций... 🎬");
            
            const canvas = canvasRef.current;
            if (canvas) {
              canvas.width = 840;
              canvas.height = 473;
            }

            isVideoInitializedRef.current = true;
          },
          onStateChange: (event) => {
            console.log('Состояние плеера изменилось:', event);
            
            if (event.playerState.ENDED) {
              console.log('Видео завершено');
              if (onProcessingComplete) {
                onProcessingComplete();
              }
            }
          }
        }
      };

      rt.Player('rutube-player-container', playerConfig);
      rutubePlayerRef.current = rt;
      
      console.log('Rutube Player инициализирован с videoId:', videoId);
      
    } catch (error) {
      console.error('Ошибка инициализации Rutube Player:', error);
      setError(`Ошибка инициализации плеера: ${error.message}`);
    }
  }, [videoUrl, extractVideoId, onProcessingComplete]);

  const scaleCoordinates = useCallback((box, displayWidth, displayHeight) => {
    const originalWidth = 1920;
    const originalHeight = 1080;
    
    const scaleX = displayWidth / originalWidth;
    const scaleY = displayHeight / originalHeight;
    
    return [
      box[0] * scaleX,
      box[1] * scaleY,
      box[2] * scaleX,
      box[3] * scaleY
    ];
  }, []);

  const drawAnnotations = useCallback(() => {
    try {
      const canvas = canvasRef.current;
      if (!canvas || !currentAnnotations.length) return;

      const context = canvas.getContext("2d");
      if (!context) return;

      context.clearRect(0, 0, canvas.width, canvas.height);

      const displayWidth = canvas.offsetWidth;
      const displayHeight = canvas.offsetHeight;

      currentAnnotations.forEach((annotation) => {
        if (annotation?.box && Array.isArray(annotation.box) && annotation.box.length >= 4) {
          const scaledBox = scaleCoordinates(annotation.box, displayWidth, displayHeight);
          const [x1, y1, x2, y2] = scaledBox;
          const width = x2 - x1;
          const height = y2 - y1;

          const colors = ['#4B8BFC', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'];
          const color = colors[annotation.track_id % colors.length];

          context.strokeStyle = color;
          context.lineWidth = 3;
          context.setLineDash([8, 4]);
          context.strokeRect(x1, y1, width, height);

          context.fillStyle = color;
          context.fillRect(x1, y1 - 30, Math.max(width, 100), 30);

          context.fillStyle = 'white';
          context.font = 'bold 14px Montserrat, sans-serif';
          context.textAlign = 'left';
          context.fillText(
            `ID: ${annotation.track_id} (${(annotation.confidence * 100).toFixed(1)}%)`,
            x1 + 5,
            y1 - 10
          );

          if (annotation.recognized_text) {
            context.fillStyle = 'rgba(75, 139, 252, 0.9)';
            context.fillRect(x1, y2, width, 25);
            context.fillStyle = 'white';
            context.font = '12px Montserrat, sans-serif';
            context.fillText(annotation.recognized_text, x1 + 5, y2 + 17);
          }
        }
      });
    } catch (error) {
      console.warn("Error drawing annotations:", error);
    }
  }, [currentAnnotations, scaleCoordinates]);

  const processFrameData = useCallback((frameData) => {
    console.log('Processing frame data:', frameData);
    
    setCurrentFrameData(frameData);
    setCurrentAnnotations(frameData.tracked_objects || []);
    
    const newTexts = frameData.tracked_objects
      ?.filter(obj => obj.recognized_text && obj.recognized_text.trim())
      ?.map(obj => obj.recognized_text.trim()) || [];
    
    if (newTexts.length > 0) {
      setRecognizedTexts(prev => [...prev, ...newTexts].slice(-10));
      
      setUniqueTexts(prevUnique => {
        const updatedUnique = new Set(prevUnique);
        const newUniqueTexts = [];
        
        newTexts.forEach(text => {
          if (!updatedUnique.has(text)) {
            updatedUnique.add(text);
            newUniqueTexts.push(text);
          }
        });
        
        if (newUniqueTexts.length > 0) {
          setAllFoundTexts(prev => [...prev, ...newUniqueTexts]);
        }
        
        return updatedUnique;
      });
    }
  }, []);

  // Функция для показа конкретного кадра
  const showFrame = useCallback((frameIndex) => {
    if (frameIndex < 0 || frameIndex >= frameQueueRef.current.length) {
      return;
    }

    const frameData = frameQueueRef.current[frameIndex];
    if (!frameData) return;

    console.log(`Показываем кадр ${frameIndex}, время: ${frameData.timestamp}s`);
    
    // Обновляем аннотации
    processFrameData(frameData);
    currentFrameIndexRef.current = frameIndex;
    
    // Позиционируем видео на нужное время
    if (rutubePlayerRef.current && isVideoInitializedRef.current) {
      rutubePlayerRef.current.seekTo({ time: frameData.timestamp });
    }
    
    setWaitingForNextFrame(false);
  }, [processFrameData]);

  useEffect(() => {
    if (!videoUrl) return;

    setStatusMessage("Загрузка видео...");
    
    setTimeout(() => {
      initializeRutubePlayer();
    }, 500);

    return () => {
      if (rutubePlayerRef.current) {
        try {
          rutubePlayerRef.current.stop();
        } catch (error) {
          console.warn('Ошибка остановки плеера:', error);
        }
      }
    };
  }, [videoUrl, initializeRutubePlayer]);

  // Основная логика: при получении новой аннотации показываем следующий кадр
  useEffect(() => {
    if (frameDataStream && !isProcessingNewFrameRef.current) {
      isProcessingNewFrameRef.current = true;
      
      console.log('Получен новый кадр:', frameDataStream);
      frameQueueRef.current.push(frameDataStream);
      
      if (playerReady && isVideoInitializedRef.current) {
        const newFrameIndex = frameQueueRef.current.length - 1;
        
        // Показываем новый кадр сразу же
        showFrame(newFrameIndex);
        setStatusMessage("Синхронизация с аннотациями ⚡");
      }
      
      // Сбрасываем флаг через небольшую задержку
      setTimeout(() => {
        isProcessingNewFrameRef.current = false;
      }, 10);
    }
  }, [frameDataStream, playerReady, showFrame]);

  useEffect(() => {
    drawAnnotations();
  }, [drawAnnotations]);

  if (error) {
    return (
      <Box
        maxW="900px"
        width="100%"
        mx="auto"
        mt={8}
        css={{
          animation: `${fadeInUp} 0.6s ease-out`
        }}
      >
        <Alert 
          status="error" 
          borderRadius="24px"
          p={8}
          bg="linear-gradient(135deg, #fee 0%, #fdd 100%)"
          border="2px solid #f56565"
          boxShadow="0 10px 40px rgba(245, 101, 101, 0.2)"
        >
          <AlertIcon color="#f56565" />
          <Box>
            <Text 
              fontWeight="bold" 
              fontSize="lg"
              fontFamily="Montserrat, sans-serif"
              color="#c53030"
            >
              Ошибка плеера 😞
            </Text>
            <Text 
              fontFamily="Montserrat, sans-serif"
              color="#744210"
              mt={2}
            >
              {error}
            </Text>
          </Box>
        </Alert>
      </Box>
    );
  }

  return (
    <Box 
      maxW="900px" 
      width="100%" 
      mx="auto" 
      mt={8}
      css={{
        opacity: 0,
        animation: `${fadeInUp} 0.8s ease-out 0.3s forwards`
      }}
    >
      <Box
        position="relative"
        bg="white"
        borderRadius="24px"
        p={{ base: "20px", sm: "30px" }}
        boxShadow="0 20px 60px rgba(0, 0, 0, 0.1)"
        css={{
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 25px 70px rgba(0, 0, 0, 0.15)',
            transition: 'all 0.3s ease'
          }
        }}
      >
        {!playerReady && (
          <Flex
            position="absolute"
            top="0"
            left="0"
            right="0"
            bottom="0"
            bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
            color="white"
            align="center"
            justify="center"
            flexDirection="column"
            zIndex="2"
            borderRadius="24px"
            css={{
              animation: `${scaleIn} 0.5s ease-out`
            }}
          >
            <Box
              css={{
                animation: `${pulse} 2s infinite`
              }}
            >
              <Spinner 
                size="xl" 
                mb={6}
                thickness="4px"
                speed="0.8s"
                color="white"
              />
            </Box>
            <Text
              fontFamily="Montserrat, sans-serif"
              fontWeight="600"
              fontSize="18px"
              textAlign="center"
              css={{
                animation: `${fadeInUp} 0.6s ease-out 0.2s both`
              }}
            >
              {statusMessage}
            </Text>
          </Flex>
        )}

        <Box
          position="relative"
          borderRadius="20px"
          overflow="hidden"
          css={{
            border: playerReady ? '3px solid #4B8BFC' : '3px solid #E2E8F0',
            animation: playerReady ? `${glowBorder} 3s ease-in-out infinite` : 'none',
            transition: 'all 0.3s ease'
          }}
        >
          <Box 
            position="relative"
            width="100%"
            height="473px"
            bg="#000"
          >
            <Box
              id="rutube-player-container"
              width="100%"
              height="100%"
              style={{
                borderRadius: '17px',
                overflow: 'hidden'
              }}
            />
            
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                zIndex: 10,
                borderRadius: '17px'
              }}
            />

            {waitingForNextFrame && playerReady && (
              <Box
                position="absolute"
                top="20px"
                right="20px"
                bg="rgba(255, 193, 7, 0.9)"
                color="white"
                p={2}
                borderRadius="8px"
                fontSize="sm"
                fontWeight="600"
                zIndex={15}
                css={{
                  animation: `${pulse} 1s infinite`
                }}
              >
                ⏳ Ожидание аннотации...
              </Box>
            )}

            {currentFrameData && playerReady && (
              <Box
                position="absolute"
                top="20px"
                left="20px"
                bg="rgba(75, 139, 252, 0.9)"
                color="white"
                p={2}
                borderRadius="8px"
                fontSize="sm"
                fontWeight="600"
                zIndex={15}
              >
                📹 Кадр: {currentFrameData.frame_number} | 🕐 {currentFrameData.timestamp.toFixed(2)}s
              </Box>
            )}
          </Box>
        </Box>

        {playerReady && (
          <Box 
            mt={6} 
            p={6} 
            bg="linear-gradient(135deg, #f6f9fc 0%, #e9f4ff 100%)"
            borderRadius="20px"
            border="1px solid #E2E8F0"
            css={{
              animation: `${fadeInUp} 0.6s ease-out 0.5s both`
            }}
          >
            <Flex direction="column" gap={4}>
              <Flex 
                direction={{ base: "column", md: "row" }}
                align={{ base: "flex-start", md: "center" }}
                justify="space-between"
                gap={3}
              >
                <Text 
                  fontSize="lg" 
                  fontWeight="700"
                  color="#023BA3"
                  fontFamily="Montserrat, sans-serif"
                  display="flex"
                  alignItems="center"
                  gap={2}
                >
                  🎬 Видео ID: {extractVideoId(videoUrl)}
                </Text>
                
                <Text 
                  fontSize="sm" 
                  fontWeight="600"
                  color="#667eea"
                  fontFamily="Montserrat, sans-serif"
                >
                  📝 Найдено уникальных текстов: {allFoundTexts.length}
                </Text>
              </Flex>

              {/* Секция с накопленным уникальным текстом */}
              {allFoundTexts.length > 0 && (
                <Box
                  bg="white"
                  borderRadius="12px"
                  border="2px solid #667eea"
                  p={4}
                  maxHeight="200px"
                  overflowY="auto"
                  css={{
                    '&::-webkit-scrollbar': {
                      width: '6px',
                    },
                    '&::-webkit-scrollbar-track': {
                      background: '#f1f1f1',
                      borderRadius: '3px',
                    },
                    '&::-webkit-scrollbar-thumb': {
                      background: '#667eea',
                      borderRadius: '3px',
                    },
                    '&::-webkit-scrollbar-thumb:hover': {
                      background: '#4B8BFC',
                    },
                  }}
                >
                  <Text 
                    fontWeight="600" 
                    color="#023BA3" 
                    mb={3}
                    fontSize="sm"
                    display="flex"
                    alignItems="center"
                    gap={2}
                  >
                    📚 Весь найденный текст:
                  </Text>
                  
                  <Flex direction="column" gap={2}>
                    {allFoundTexts.map((text, index) => (
                      <Box
                        key={`unique-${index}`}
                        p={2}
                        bg="linear-gradient(135deg, #f8faff 0%, #e6f3ff 100%)"
                        borderRadius="8px"
                        border="1px solid #E2E8F0"
                        css={{
                          '&:hover': {
                            transform: 'translateX(2px)',
                            boxShadow: '0 2px 8px rgba(75, 139, 252, 0.15)',
                            transition: 'all 0.2s ease'
                          }
                        }}
                      >
                        <Text
                          fontFamily="Montserrat, sans-serif"
                          fontWeight="500"
                          fontSize="14px"
                          color="#2D3748"
                          wordBreak="break-word"
                        >
                          <Text as="span" fontWeight="600" color="#667eea">
                            {index + 1}.
                          </Text>{" "}
                          {text}
                        </Text>
                      </Box>
                    ))}
                  </Flex>
                </Box>
              )}

              {/* Статистика */}
              <Flex
                gap={3}
                fontSize="sm"
                color="#4A5568"
                fontFamily="Montserrat, sans-serif"
                justifyContent="space-between"
              >
                <Box
                  p={3}
                  bg="white"
                  borderRadius="12px"
                  border="1px solid #E2E8F0"
                  textAlign="center"
                >
                  <Text fontWeight="600" color="#023BA3" mb={1}>📊 Статус:</Text>
                  <Text fontSize="xs">
                    ⚡ Синхронизация
                  </Text>
                </Box>
                
                <Box
                  p={3}
                  bg="white"
                  borderRadius="12px"
                  border="1px solid #E2E8F0"
                  textAlign="center"
                >
                  <Text fontWeight="600" color="#023BA3" mb={1}>🎯 Объекты:</Text>
                  <Text fontSize="xs">{currentAnnotations.length}</Text>
                </Box>
                
                <Box
                  p={3}
                  bg="white"
                  borderRadius="12px"
                  border="1px solid #E2E8F0"
                  textAlign="center"
                >
                  <Text fontWeight="600" color="#023BA3" mb={1}>⏱️ Кадры:</Text>
                  <Text fontSize="xs">
                    {currentFrameIndexRef.current + 1}/{frameQueueRef.current.length}
                  </Text>
                </Box>

                <Box
                  p={3}
                  bg="white"
                  borderRadius="12px"
                  border="1px solid #E2E8F0"
                  textAlign="center"
                >
                  <Text fontWeight="600" color="#023BA3" mb={1}>📚 Тексты:</Text>
                  <Text fontSize="xs">{allFoundTexts.length}</Text>
                </Box>
              </Flex>
            </Flex>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default VideoPlayerWithAnnotations;