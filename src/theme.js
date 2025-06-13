import { extendTheme } from "@chakra-ui/react";

const theme = extendTheme({
  config: {
    initialColorMode: "light",
    useSystemColorMode: true,
  },
  fonts: {
    body: `Montserrat, sans-serif`,
  },

  components: {
    Tooltip: {
      baseStyle: {
        background: "rgba(0, 0, 0, 0.5);",
        padding: "5px",
        fontSize: "14px !important",
      },
    },
    Button: {
      variants: {
        menu_yellow: {
          border: "2px solid",
          borderColor: "main_yellow",
          borderRadius: "0",
          background: "transparent",
          // width: '100%',
          _hover: {
            backgroundColor: "main_yellow",
          },
        },
        menu_red: {
          border: "2px solid",
          borderColor: "main_red",
          borderRadius: "0",
          background: "transparent",
          _hover: {
            backgroundColor: "main_red",
          },
        },
      },
    },
    Link: {
      variants: {
        light_gray: {
          fontColor: "light_dark",
          fontSize: ["12px", "13px", "14px"],
        },
      },
    },
    NavLink: {
      variants: {
        light_gray: {
          fontColor: "light_dark",
          fontSize: ["12px", "13px", "14px"],
        },
      },
    },
    Text: {
      variants: {
        light_gray: {
          fontColor: "light_dark",
          fontSize: ["12px", "13px", "14px"],
        },
      },
    },
    HStack: {
      variants: {
        menu_yellow_hover: {
          height: "100%",
          _hover: {
            borderBottom: "2px solid #FFBF00",
          },
        },
      },
    },
  },
  colors: {
    menu_gray: "#CCC3C2",
    main_dark: "#333",
    light_dark: "#666",
    main_yellow: "#FFBF00",
    main_red: "#FF0F00",
    menu_white: "#F8F8F8",
    date_gray: "#A9A9A9",
  },
});

export default theme;
