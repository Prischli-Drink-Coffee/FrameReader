import { 
  Box, 
  Flex, 
  Text, 
  Button, 
  FormControl, 
  Input, 
  Spinner,
  useColorModeValue
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import { useBreakpointValue } from "@chakra-ui/react";
import { useState, useEffect } from "react";
import useWindowDimensions from "../hooks/window_dimensions";

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

const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(75, 139, 252, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(75, 139, 252, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(75, 139, 252, 0);
  }
`;

const shimmer = keyframes`
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
`;

const ContentSection = ({ onProcessVideo, processingStatus }) => {
  const { height } = useWindowDimensions();
  const [url, setUrl] = useState("");
  const [isVisible, setIsVisible] = useState(false);
  const [currentFeature, setCurrentFeature] = useState(0);
  const isLoading = processingStatus === "processing";

  const features = [
    "🎥 Детекция с помощью Yolo? Скукота...",
    "🔍 OCR Donut подходит для документов, но мы подумали, а что если...",
    "📊 Никакой детальной аналитики, как я вообще плеер сюда прикрутил, лол",
    "⚡ Что там по скорости? Бабка быстрее дорогу перейдет...",
    "📈 А если серьезно, то это просто для галочки, не ждите чудес",
    "💡 Но мы все равно будем делать вид, что это круто, окей?",
    "🚀 И не забудьте, что это просто демо, а не реальный продукт",
    "🛠️ Если хотите что-то серьезное, то лучше не сюда",
    "💬 И да, не забудьте оставить отзыв, мы же все равно ничего не исправим",
    "🎉 Спасибо, что выбрали наш продукт, видимо у вас не было выбора",
    "💬 И не забудьте поделиться с друзьями, пусть тоже посмеются",
    "🎊 А если вам не понравилось, то мне все равно за это не платят",
    "🤔 И да, мы знаем, что это не лучший продукт, и мы и не старались так-то",
    "💖 Не пишите на почту dfvolkhin@edu.hse.ru, там все равно никто не ответит",
  ];

  useEffect(() => {
    setIsVisible(true);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [features.length]);

  const handleProcess = () => {
    if (url.trim()) {
      onProcessVideo(url);
    }
  };

  return (
    <Box 
      maxW="900px" 
      width="100%" 
      position="relative" 
      bg="white"
      p={{ base: "30px", sm: "40px" }}
      borderRadius="24px"
      boxShadow="0 20px 60px rgba(0, 0, 0, 0.1)"
      css={{
        animation: `${fadeInUp} 0.8s ease-out`,
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: '0 25px 70px rgba(0, 0, 0, 0.15)',
          transition: 'all 0.3s ease'
        }
      }}
    >
      <Flex direction="column" align="flex-start" gap={height * 0.04}>
        <Box position="relative" overflow="hidden">
          <Text
            width="100%"
            fontFamily="Montserrat, sans-serif"
            fontWeight="800"
            fontSize={{ base: "28px", sm: "36px", md: "42px" }}
            lineHeight="1.2"
            bgGradient="linear(to-r, #023BA3, #4B8BFC, #667eea)"
            bgClip="text"
            css={{
              opacity: 0,
              animation: `${fadeInUp} 0.8s ease-out 0.2s forwards`
            }}
          >
            FrameReader
          </Text>
          <Text
            mt="10px"
            fontFamily="Montserrat, sans-serif"
            fontWeight="600"
            fontSize={{ base: "18px", sm: "20px" }}
            color="#023BA3"
            css={{
              opacity: 0,
              animation: `${fadeInUp} 0.8s ease-out 0.6s forwards`
            }}
          >
            Это что Искусственный интеллект? Опять?
          </Text>
        </Box>

        <Box
          width="100%"
          css={{
            opacity: 0,
            animation: `${fadeInUp} 0.8s ease-out 1s forwards`
          }}
        >
          <Text
            fontFamily="Montserrat, sans-serif"
            fontWeight="500"
            fontSize={{ base: "16px", sm: "18px" }}
            lineHeight="1.6"
            color="#4A5568"
            mb="20px"
          >
            Революционная (ха-ха-ха-ха) система для автоматического распознавания и извлечения текста 
            из видеоматериалов. Загрузите ссылку на видео с Rutube и получите детальный 
            анализ всего текстового содержимого, но за качество не отвечаем.
          </Text>

          <Box
            height="60px"
            display="flex"
            alignItems="center"
            bg="linear-gradient(135deg, #f6f9fc 0%, #e9f4ff 100%)"
            borderRadius="12px"
            p="15px"
            border="1px solid #E2E8F0"
          >
            <Text
              fontFamily="Montserrat, sans-serif"
              fontWeight="600"
              fontSize="16px"
              color="#2D3748"
              key={currentFeature}
              css={{
                animation: `${fadeInUp} 0.5s ease-out`
              }}
            >
              {features[currentFeature]}
            </Text>
          </Box>
        </Box>

        <Flex 
          direction={{ base: "column", sm: "row" }} 
          align={{ base: "stretch", sm: "flex-end" }}
          gap="20px" 
          w="100%"
          css={{
            opacity: 0,
            animation: `${fadeInUp} 0.8s ease-out 1.4s forwards`
          }}
        >
          <FormControl id="video-url" isRequired flex="1">
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              height="70px"
              border="3px solid #4B8BFC"
              borderRadius="20px"
              placeholder="🔗 Вставьте ссылку на видео Rutube..."
              paddingLeft="25px"
              paddingRight="25px"
              bg="white"
              fontSize="16px"
              fontFamily="Montserrat, sans-serif"
              _placeholder={{
                fontFamily: "Montserrat, sans-serif",
                fontWeight: "500",
                fontSize: "16px",
                color: "#A0AEC0",
              }}
              css={{
                '&:focus': {
                  outline: 'none',
                  animation: `${pulse} 2s infinite`,
                  borderColor: '#023BA3'
                },
                '&:hover': {
                  transform: 'translateY(-2px)',
                  transition: 'all 0.3s ease'
                }
              }}
            />
          </FormControl>

          <Button
            onClick={handleProcess}
            width={{ base: "100%", sm: "250px" }}
            height="70px"
            background="linear-gradient(135deg, #4B8BFC 0%, #023BA3 100%)"
            borderRadius="20px"
            fontFamily="Montserrat, sans-serif"
            fontWeight="700"
            fontSize="18px"
            color="white"
            position="relative"
            overflow="hidden"
            disabled={isLoading || !url.trim()}
            css={{
              '&:hover:not(:disabled)': {
                transform: 'translateY(-3px)',
                boxShadow: '0 15px 35px rgba(75, 139, 252, 0.4)'
              },
              '&:active:not(:disabled)': {
                transform: 'translateY(-1px)'
              },
              '&:disabled': {
                opacity: 0.7,
                cursor: 'not-allowed'
              }
            }}
          >
            {isLoading ? (
              <Flex align="center" gap="10px">
                <Spinner size="sm" color="white" />
                <Text>Обработка...</Text>
              </Flex>
            ) : (
              <Flex align="center" gap="10px">
                <Text>🚀 Анализировать</Text>
              </Flex>
            )}
          </Button>
        </Flex>

        {processingStatus === "processing" && (
          <Box
            width="100%"
            mt="20px"
            css={{
              animation: `${fadeInUp} 0.5s ease-out`
            }}
          >
            <Text
              fontFamily="Montserrat, sans-serif"
              fontWeight="600"
              fontSize="16px"
              color="#4B8BFC"
              textAlign="center"
              mb="10px"
            >
              Магия искусственного интеллекта в действии...
            </Text>
            <Box
              width="100%"
              height="4px"
              bg="#E2E8F0"
              borderRadius="2px"
              overflow="hidden"
            >
              <Box
                height="100%"
                bg="linear-gradient(90deg, #4B8BFC, #667eea, #4B8BFC)"
                backgroundSize="200% 200%"
                css={{
                  animation: `${shimmer} 2s ease-in-out infinite`
                }}
                borderRadius="2px"
              />
            </Box>
          </Box>
        )}
      </Flex>
    </Box>
  );
};

export default ContentSection;