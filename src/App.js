import { createHashRouter, RouterProvider, Navigate } from "react-router-dom";
import Layout from "./Layout";
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
            }
        ],
    },
]);


function App() {
    return <RouterProvider router={router} />;
}

export default App;
