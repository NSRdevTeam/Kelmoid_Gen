import { BrowserRouter, Routes, Route } from "react-router-dom";
import Login from "./src/components/auth/Login";
import Index from "./index"; 

const RootRouter = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Auth */}
        <Route path="/" element={<Login />} />

        {/* Kelmoid Genesis – Text → CAD */}
        <Route path="/app" element={<Index />} />
      </Routes>
    </BrowserRouter>
  );
};

export default RootRouter;
