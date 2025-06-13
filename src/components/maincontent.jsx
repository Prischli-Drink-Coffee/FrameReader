import { Box, Flex, Text, Button, FormControl, Input, Spinner } from "@chakra-ui/react";
import { useBreakpointValue } from "@chakra-ui/react";
import { useState } from "react";
import useWindowDimensions from "../hooks/window_dimensions";

const ContentSection = ({ onFetch }) => {
  const { height } = useWindowDimensions();
  const [url, setUrl] = useState(""); // Состояние для URL
  const [isLoading, setIsLoading] = useState(false); // Состояние для индикатора загрузки
  const buttonText = useBreakpointValue({
    base: "What about it?",
    sm: "About it?",
  });

  const handleFetch = () => {
    if (url.trim()) {
      setIsLoading(true); // Устанавливаем состояние загрузки в true
      onFetch(url).finally(() => setIsLoading(false)); // После завершения обработки выключаем индикатор загрузки
    }
  };

  return (
    <Box maxW="840px" width="100%" position="relative" bg="#ffffff" p={{ base: "20px", sm: "30px" }}>
      <Flex direction="column" align="flex-start" gap={height * 0.045}>
        <Text
          width={{ base: "100%", sm: "100%" }}
          fontFamily="Montserrat"
          fontWeight="700"
          fontSize={{ base: "24px", sm: "32px" }}
          lineHeight="46px"
          color="#023BA3"
        >
          Too lazy to watch the video? Let's say what it's about
        </Text>
        <Text
          width={{ base: "100%", sm: "100%" }}
          fontFamily="Montserrat"
          fontWeight="500"
          fontSize={{ base: "16px", sm: "18px" }}
          lineHeight="22px"
          color="#1D1D1D"
        >
          NaRuTagAI - allows you to automate the generation of hierarchical tags for videos using multimodal
          artificial intelligence. The system uses visual, audio, and text information to make recommendations
          for the most suitable tags.
        </Text>
        <Flex direction={{ base: "column", sm: "row" }} align="flex-start" gap="20px" w="100%">
          <FormControl id="URL" isRequired>
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)} // Управляем вводом URL
              width={{ base: "100%", sm: "100%" }}
              height="60px"
              border="4px solid #4B8BFC"
              borderRadius="16px"
              placeholder="Enter URL here"
              paddingLeft="20px"
              bg="#FFFFFF"
              _placeholder={{
                fontFamily: "Montserrat",
                fontWeight: "500",
                fontSize: "18px",
                lineHeight: "22px",
                color: "#1D1D1D",
              }}
            />
          </FormControl>
          <Button
            onClick={handleFetch} // Обработчик клика
            width={{ base: "100%", sm: "240px" }}
            height="60px"
            background="#4B8BFC"
            borderRadius="16px"
            fontFamily="Montserrat"
            fontWeight="700"
            fontSize="18px"
            lineHeight="22px"
            color="#FFFFFF"
            _hover={{ background: "#376fcb" }}
            disabled={isLoading} // Отключаем кнопку, пока идет процесс загрузки
          >
            {isLoading ? (
              <Spinner size="sm" color="white" /> // Показываем спиннер при загрузке
            ) : (
              buttonText
            )}
          </Button>
        </Flex>
      </Flex>
    </Box>
  );
};

export default ContentSection;
