import { HStack, Text, Flex } from "@chakra-ui/react";

const Footer = () => {
  return (
    <Flex
      backgroundColor="#4B8BFC"
      width="100%"
      minHeight={["60px", "70px", "80px", "90px", "100px"]} // Адаптивная минимальная высота
      padding={["10px", "12px", "15px", "20px", "25px"]} // Адаптивные отступы
      justifyContent="center"
    >
      <HStack
        justify={["center", "space-between"]} // На мобильных — по центру, на большем экране — space-between
        width={["100%", "50%"]} // На мобильных — 100%, на более широких экранах — 50%
        spacing={["10px", "20px"]} // Добавляем разные интервалы между элементами
        textAlign={["center", "left"]} // На мобильных — по центру, на большем — слева
        wrap="wrap" // Обеспечивает перенос на маленьких экранах
      >
        <Text
          color="#FFFFFF"
          fontFamily="Montserrat"
          fontSize={["14px", "16px", "18px"]} // Адаптивный размер шрифта
          lineHeight="22px"
          fontWeight="500"
          width="auto" // Убираем фиксированную ширину для текста
          textAlign={["center", "right"]} // Выравнивание для мобильных и больших экранов
        >
          Команда «Придумать название»
        </Text>

        <HStack
          onClick={() => {
            window.scrollTo(0, 0);
          }}
          spacing={["5px", "10px"]} // Добавляем адаптивный промежуток между элементами
          justify="center" // Выравниваем по центру для мобильных
        >
          <Text
            color="#FFFFFF"
            fontFamily="Montserrat"
            fontSize={["14px", "16px", "18px"]} // Адаптивный размер шрифта
            lineHeight="22px"
            fontWeight="500"
            width="auto" // Убираем фиксированную ширину
            textAlign={["center", "left"]} // Для мобильных — по центру, для больших экранов — слева
          >
            Проект по машинному обучению МТС AI
          </Text>
        </HStack>
      </HStack>
    </Flex>
  );
};

export default Footer;
