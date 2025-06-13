import React from 'react';
import { Box, Flex, Text, VStack } from "@chakra-ui/react";

const TagCloud = () => {
    return (
        <Box
            position="absolute"
            width="700px"
            height="600px"
            left="900px"
            p="4"
            bg="transpose"
        >

        <VStack spacing="41" mt="100px">
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

            {/* Блок с тегами */}
            <Flex
                display="flex"
                flexDirection="row"
                flexWrap="wrap"
                justifyContent="center"
                alignItems="center"
                alignContent="center"
                gap="10px"
                width="417px"
            >
                {/* Список тегов */}
                {[
                    { text: "Artificial Intelligence System", width: "403px" },
                    { text: "Vector search", width: "215px" },
                    { text: "Computer vision", width: "264px" },
                    { text: "Classification", width: "196px" },
                    { text: "LLM", width: "79px" },
                ].map((tag, index) => (
                    <Flex
                        key={index}
                        display="flex"
                        flexDirection="row"
                        justifyContent="center"
                        alignItems="center"
                        padding="19px 20px"
                        gap="10px"
                        width={tag.width}
                        height="40px"
                        bg="#4B8BFC"
                        borderRadius="12px"
                    >
                        <Text
                            fontFamily="Montserrat"
                            fontWeight="700"
                            fontSize="18px"
                            lineHeight="22px"
                            color="#FFFFFF"
                        >
                            {tag.text}
                        </Text>
                    </Flex>
                ))}
            </Flex>
        </VStack>
        </Box>
    );
};

export default TagCloud;
