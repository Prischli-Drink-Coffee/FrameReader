// import React, { useEffect } from "react";
import React from "react";
import { VStack, Divider } from "@chakra-ui/react";
import Header from "./components/header/header";
import { Outlet } from "react-router-dom";
import Footer from "./components/footer";
// import { useCookies } from "react-cookie";
// import UserService from "./API/services/user_service";
// import { deleteUser } from "./API/helper/userCookie";

function Layout() {
  // const [cookie, setCookie] = useCookies();
  // const getUser = async () => {
  //   try {
  //     const response = await UserService.me();
  // 
  //     if (response.data.userEmail !== cookie.userEmail) {
  //       setCookie("userEmail", response.data.userEmail);
  //     }
  //   } catch (e) {
  //     if (e.response.status === 401) {
  //       deleteUser();
  //     }
  //   }
  // };
  // useEffect(() => {
  //   getUser();
  // }, [cookie]);

  return (
    <VStack
      backgroundColor="menu_white"
      width="100%"
      minH="100vh"
      spacing={0}
      align="stretch"
    >
      <Header />
      <Outlet />
      <Footer />
    </VStack>
  );
}

export default Layout;
