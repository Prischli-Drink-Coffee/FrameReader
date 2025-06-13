import { Flex, VStack, Divider } from "@chakra-ui/react";
import React from "react";
import LoginForm from "../components/logform";
import Slogan from "../components/slogan";
import UserService from "../API/services/user_service";
import { setUser } from "../API/helper/userCookie";
import { useNavigate } from "react-router";
import useWindowDimensions from "../hooks/window_dimensions";


const SignInPage = () => {
    const navigate = useNavigate();
    const { width } = useWindowDimensions();

  const validate = (values) => {
    const errors = {};

    if (!values.login) {
      errors.login = "Required";
    } else if (values.login.length > 15) {
      errors.login = "Must be 15 characters or less";
    }
    if (!values.password) {
      errors.password = "Required";
    }

    return errors;
    };

  const signIn = async (values) => {
    try {
      const response = await UserService.signIn(values.login, values.password);
      const me = await UserService.me(response.data.token);
      setUser({ ...me.data, token: response.data.token });
      alert("Вы успешно вошли в систему");
      navigate("/main");
    } catch (error) {
      console.error("Error signIn:", error);
      alert("Ошибка в данных!");
    }
  };

    return (
        <VStack minH="100vh" align="center" justify="center" bg="#ffffff">
            <Flex
                position="relative"
                width={ width }
                height="auto"
                align="center"
                justify="center"
                gap="20px"
            >

                {/* Slogan слева*/}
                <Slogan />

                {/* Разделительная линия */}
                <Divider
                    orientation="vertical"
                    border="1px solid rgba(75, 187, 252, 0.6)"
                    height="600px"
                />

                {/* Форма авторизации справа с передачей функции signIn */}
                <LoginForm onSubmit={signIn} />

            </Flex>
        </VStack>
    );
};

export default SignInPage;
