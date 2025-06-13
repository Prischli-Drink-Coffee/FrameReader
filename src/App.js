import { createHashRouter, RouterProvider, Navigate } from "react-router-dom";
import Layout from "./Layout";
// import SignInPage from "./pages/sign_in_page";
// import SignUpPage from "./pages/sign_up_page";
import MainPage from "./pages/main_page";
import NotFoundPage from "./pages/notfound_page";


const router = createHashRouter([
    {
        element: <Layout />,
        children: [
            {
                path: "/",
                element: <Navigate to="/main" />,
            },
            {
                path: "/main",
                element: <MainPage />,
                errorElement: <NotFoundPage />,
            },
            // {
            //     path: "/sign_up",
            //     element: <SignUpPage />,
            //     errorElement: <NotFoundPage />,
            // },
            // {
            //     path: "/sign_in",
            //     element: <SignInPage />,
            //    errorElement: <NotFoundPage />,
            // },
        ],
    },
]);


function App() {
    return <RouterProvider router={router} />;
}

export default App;
