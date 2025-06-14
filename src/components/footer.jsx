import { HStack, Text, Flex, Divider } from "@chakra-ui/react";

const Footer = () => {
  return (
    <Flex
      backgroundColor="#4B8BFC"
      width="100%"
      minHeight={["60px", "70px", "80px", "90px", "100px"]}
      padding={["10px", "12px", "15px", "20px", "25px"]}
      justifyContent="center"
      alignItems="center"
    >
      <HStack
        justify="space-between"
        align="center"
        width="100%"
        maxWidth="1200px"
        spacing={["15px", "20px", "30px"]}
        wrap={["wrap", "wrap", "nowrap"]}
      >
        <Text
          color="#FFFFFF"
          fontFamily="Montserrat, sans-serif"
          fontSize={["12px", "14px", "16px"]}
          lineHeight="1.4"
          fontWeight="500"
          textAlign={["center", "center", "left"]}
          flex="1"
        >
          Команда «Пришли пить кофе» | ФКН | ВШЭ | 2025
        </Text>

        <Divider
          orientation="vertical"
          height={["20px", "25px", "30px"]}
          borderColor="rgba(255, 255, 255, 0.3)"
          borderWidth="1px"
          display={["none", "none", "block"]}
        />

        <Text
          color="#FFFFFF"
          fontFamily="Montserrat, sans-serif"
          fontSize={["12px", "14px", "16px"]}
          lineHeight="1.4"
          fontWeight="500"
          textAlign={["center", "center", "right"]}
          flex="1"
          cursor="pointer"
          onClick={() => window.scrollTo(0, 0)}
          _hover={{
            opacity: 0.8,
            transition: "opacity 0.2s ease"
          }}
        >
          Проект «FrameReader» не спонсирован кафедрой МТС
        </Text>
      </HStack>
    </Flex>
  );
};

export default Footer;