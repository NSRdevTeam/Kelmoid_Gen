import AnimatedBackground from "./AnimatedBackground";
import BrandingPanel from "./BrandingPanel";
import LoginForm from "./LoginForm";
import KelmoidLogo from "./KelmoidLogo";
import TrustBadges from "./TrustBadges";

const Login = () => {
  return (
    <div className="min-h-screen w-full flex">
      <AnimatedBackground />
      
      {/* Left Panel - Branding (hidden on mobile) */}
      <div className="hidden lg:flex lg:w-1/2 xl:w-3/5 relative z-10">
        <BrandingPanel />
      </div>

      {/* Right Panel - Login Form */}
      <div className="w-full lg:w-1/2 xl:w-2/5 flex items-center justify-center p-6 lg:p-12 relative z-10">
        <div className="w-full max-w-md space-y-8">
          {/* Mobile Logo */}
          <div className="lg:hidden flex justify-center animate-fade-in">
            <KelmoidLogo size="lg" />
          </div>

          {/* Login Card */}
          <div className="glass-card p-8 space-y-6 animate-fade-in" style={{ animationDelay: "0.1s" }}>
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold text-foreground">
                Welcome back
              </h2>
              <p className="text-muted-foreground text-sm">
                Sign in to access your CAD workspace
              </p>
            </div>

            <LoginForm />
          </div>

          {/* Trust Badges */}
          <div className="animate-fade-in" style={{ animationDelay: "0.2s" }}>
            <TrustBadges />
          </div>

          {/* Mobile Tagline */}
          <div className="lg:hidden text-center animate-fade-in" style={{ animationDelay: "0.3s" }}>
            <p className="text-sm text-muted-foreground">
              AI-Powered CAD Intelligence for Modern Manufacturing
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
