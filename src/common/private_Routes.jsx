import { Navigate, Outlet } from "react-router-dom";

const PrivateRoutes = ({ userGroup }) => {

  let isAllowed = false;

  if (userGroup === "AUTH") {
    isAllowed = true;
  }
  return isAllowed ? <Outlet /> : <Navigate to="/main" />;
};

export default PrivateRoutes;
