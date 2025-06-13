import { Box, Flex, Text, VStack } from "@chakra-ui/react";

const TagSection = ({ video }) => {
  return (
    <Box maxW="840px" width="100%" position="relative">
      <Box position="relative" bg="#ffffff" p={{ base: "20px", sm: "40px" }}>
        {/* Заголовок */}
        <Text
          width="100%"
          fontFamily="Montserrat"
          fontWeight="700"
          fontSize={{ base: "24px", sm: "32px" }}
          lineHeight="39px"
          color="#023BA3"
          mb="20px"
        >
          As we understand, this is about...
        </Text>

        {/* Рендер иерархии */}
        <VStack
          align="flex-start"
          spacing="20px"
          width="100%"
          pl={{ base: "10px", sm: "20px" }} // Отступы для всей структуры
        >
          {Object.entries(video).map(([category, subCategories]) => (
            <Box key={category} width="100%">
              {/* Категория */}
              <Text
                fontFamily="Montserrat"
                fontWeight="700"
                fontSize={{ base: "20px", sm: "24px" }}
                lineHeight="32px"
                color="#4B8BFC"
                mb="10px"
              >
                {category}
              </Text>

              {/* Подкатегории */}
              <VStack align="flex-start" spacing="16px" pl="20px">
                {Object.entries(subCategories).map(([subCategory, tags]) => (
                  <Box key={subCategory} width="100%">
                    {/* Подкатегория */}
                    <Text
                      fontFamily="Montserrat"
                      fontWeight="600"
                      fontSize={{ base: "18px", sm: "20px" }}
                      lineHeight="28px"
                      color="#023BA3"
                      mb="8px"
                    >
                      {subCategory}
                    </Text>

                    {/* Теги */}
                    <Flex
                      direction="row"
                      gap="10px"
                      flexWrap="wrap"
                      pl="20px" // Отступ для тегов
                    >
                      {/* Проверяем, является ли tags массивом */}
                      {(Array.isArray(tags) ? tags : []).map((tag, index) => (
                        <Box
                          key={index}
                          width="auto"
                          maxWidth="300px"
                          height="60px"
                          background="#4B8BFC"
                          borderRadius="12px"
                          display="flex"
                          alignItems="center"
                          justifyContent="center"
                          padding={{
                            base: "0px 8px",
                            sm: "0px 12px",
                          }}
                          textAlign="center"
                        >
                          <Text
                            fontFamily="Montserrat"
                            fontWeight="500"
                            fontSize={{
                              base: "12px",
                              sm: "14px",
                              md: "16px",
                              lg: "18px",
                            }}
                            lineHeight="22px"
                            color="#FFFFFF"
                          >
                            {tag}
                          </Text>
                        </Box>
                      ))}
                    </Flex>
                  </Box>
                ))}
              </VStack>
            </Box>
          ))}
        </VStack>
      </Box>
    </Box>
  );
};

export default TagSection;