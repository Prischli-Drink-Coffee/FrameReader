import { VStack, Box, Text } from "@chakra-ui/react";
import useWindowDimensions from "../hooks/window_dimensions";
import ContentSection from "../components/maincontent";
import TagSection from "../components/tagsectionmain";
import { sendInference } from "../API/services/inference_services";
import React, { useState, useEffect } from "react";


const MainPage = () => {
  const { height } = useWindowDimensions();
  const [response, setResponse] = useState(null); // Состояние для ответа от API
  const [error, setError] = useState(null); // Состояние для ошибок

  // Функция для получения тегов с сервера
  const fetchTags = async (url) => {
    try {
      const data = await sendInference(url);
      console.log("Inference response:", data);
      setResponse(data); // Сохраняем преобразованные данные в состояние
    } catch (err) {
      console.error("Failed to fetch inference data:", err);
      setError(err); // Сохраняем ошибку в состоянии
    }
  };

  // Эффект для обработки ошибки
  useEffect(() => {
    if (error) {
      // В случае ошибки устанавливаем ответ с деталями ошибки
      setResponse({
        Error: {
          Details: [
            "Failed fetch data",
            "Check video URL"
          ]
        }
      });
    }
  }, [error]); // Эффект срабатывает при изменении ошибки

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
        <ContentSection onFetch={fetchTags} />
        <Box mt={height > 600 ? height * 0.05 : "20px"}>
          {response ? (
            <TagSection video={response} />
          ) : (
            <Text fontSize="18px" color="#666">
              Enter a video URL to see tags.
            </Text>
          )}
        </Box>
      </Box>
    </VStack>
  );
};

export default MainPage;