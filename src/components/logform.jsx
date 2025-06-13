import React from 'react';
import { Box, Button, Input, Text, VStack, FormControl } from "@chakra-ui/react";

const LoginForm = () => {
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
                    color="#023BA3"
                >
                    Authorization
                </Text>

                {/* Поле ввода Email */}
                <FormControl id="email" isRequired>
                    <Input
                        width="400px"
                        height="60px"
                        border="4px solid #4B8BFC"
                        borderRadius="16px"
                        placeholder="Email"
                        bg="#FFFFFF"
                        left="50%"
                        transform="translateX(-50%)"
                    />
                </FormControl>

                {/* Поле ввода пароля */}
                <FormControl id="password" isRequired>
                    <Input
                        type="password"
                        width="400px"
                        height="60px"
                        border="4px solid #4B8BFC"
                        borderRadius="16px"
                        placeholder="Password"
                        bg="#FFFFFF"
                        left="50%"
                        transform="translateX(-50%)"
                    />
                </FormControl>

                {/* Кнопка отправки */}
                <Button
                    width="180px"
                    height="60px"
                    bg="#4B8BFC"
                    borderRadius="16px"
                    fontFamily="Montserrat"
                    fontWeight="700"
                    fontSize="18px"
                    color="#FFFFFF"
                    _hover={{ bg: "#3a6fdc" }}
                >
                    Log
                </Button>
            </VStack>
        </Box>
    );
};

export default LoginForm;
