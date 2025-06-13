import React from "react";
import logo from "./../../images/logo.svg";
import {
  Flex,
  Image,
  HStack,
  Text,
  Box
} from "@chakra-ui/react";
import tr_logo from "../../images/transparent_logo.svg";
// import { useNavigate } from "react-router";

const Header = () => {
  // const btnRef = useRef();
  // const navigate = useNavigate();

  return (
    <Flex
      as="header"
      width="100%"
      height={{ base: "80px", sm: "100px" }} // Адаптивная высота header
      bg="#FFFFFF"
      align="center"
      justify={{ base: "center", sm: "flex-start" }} // Для мобильных по центру, для десктопа влево
      p={{ base: "10px", sm: "20px" }} // Адаптивные отступы
      flexDirection={{ base: "column", sm: "row" }} // Для маленьких экранов элементы по вертикали, для больших - по горизонтали
    >
      <Box
        zIndex={1} // Делаем его "под" обычным логотипом
        position="absolute"
      >
        <Image
          src={tr_logo}
          boxSize={{ base: "150px", sm: "300px" }} // Адаптивный размер логотипа
          alt="Logo"
        />
      </Box>
      {/* Логотип и название NaRuTagAI */}
      <HStack
        bg="transparent"
        ml={{ base: "none", sm: "100px" }}
        zIndex={2} // Это поверх прозрачного логотипа
      >
        <Image
          src={logo}
          boxSize={{ base: "20px", sm: "28px" }} // Адаптивный размер логотипа
          alt="Logo"
        />
        <Text
          fontFamily="Montserrat"
          fontWeight="700"
          fontSize={{ base: "16px", sm: "18px" }} // Адаптивный размер шрифта
          lineHeight="22px"
          color="#4B8BFC"
        >
          NaRuTagAI
        </Text>
      </HStack>

      {/* Кнопки навигации на больших экранах
      <Flex
        display={{ base: "none", sm: "flex" }} // Скрываем кнопки на мобильных устройствах
        flexDirection="row"
        justifyContent="flex-end"
        alignItems="center"
        gap="20px"
      >
        <Button
          bg="#4B8BFC"
          color="#fff"
          fontWeight="700"
          fontSize="16px"
          borderRadius="8px"
          _hover={{ bg: "#376fcb" }}
          px="16px"
          py="8px"
        >
          Вход
        </Button>
        <Button
          bg="#fff"
          color="#4B8BFC"
          fontWeight="700"
          fontSize="16px"
          borderRadius="8px"
          _hover={{ bg: "#f0f0f0" }}
          px="16px"
          py="8px"
        >
          Регистрация
        </Button>
      </Flex>
      */}
    </Flex>
  );
};

export default Header;
