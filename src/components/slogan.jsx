import React from 'react';
import { Box, Flex, Text, VStack } from "@chakra-ui/react";

const TagCloud = () => {
    return (
        <Box
            position="absolute"
            width="700px"
            height="600px"
            left="100px"
            p="4"
            bg="transpose"
        >

        <VStack spacing="55" mt="100px">
            {/* Заголовок компонента */}
            <Text
                fontFamily="Montserrat"
                fontWeight="700"
                fontSize="32px"
                lineHeight="39px"
                color="#023BA3"
            >
               NaRuTagAI - it's about video tags
            </Text>

            {/* Информационный текст */}
            <Text
                fontFamily="Montserrat"
                fontWeight="500"
                fontSize="24px"
                lineHeight="29px"
                color="#1D1D1D"
                textAlign="center"
                maxWidth="400px"
            >
               The system will automatically identify key aspects of the video and generate a list of suitable tags
            </Text>
            
        </VStack>
        </Box>
    );
};

export default TagCloud;
