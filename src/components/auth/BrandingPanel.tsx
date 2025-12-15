import KelmoidLogo from "./KelmoidLogo";
import { Cpu, BarChart3, Zap, Layers } from "lucide-react";

const BrandingPanel = () => {
  const features = [
    {
      icon: Cpu,
      title: "AI-Generated CAD",
      description: "Create production-ready models in seconds",
    },
    {
      icon: BarChart3,
      title: "Manufacturing Analytics",
      description: "Real-time insights into your production line",
    },
    {
      icon: Zap,
      title: "Instant Processing",
      description: "Enterprise-grade speed and reliability",
    },
    {
      icon: Layers,
      title: "Multi-Format Export",
      description: "STEP, STL, IGES, and 20+ formats",
    },
  ];

  return (
    <div className="flex flex-col justify-between h-full p-8 lg:p-12">
      <div>
        <KelmoidLogo size="lg" className="animate-fade-in" />
        
        <div className="mt-12 space-y-4 animate-fade-in" style={{ animationDelay: "0.1s" }}>
          <h1 className="text-3xl lg:text-4xl xl:text-5xl font-bold leading-tight">
            AI-Powered CAD Intelligence for{" "}
            <span className="gradient-text">Modern Manufacturing</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-md">
            Transform your design workflow with generative AI. From concept to production in minutes, not months.
          </p>
        </div>
      </div>

      <div className="mt-12 grid grid-cols-1 sm:grid-cols-2 gap-4">
        {features.map((feature, index) => (
          <div
            key={index}
            className="group p-4 rounded-xl bg-secondary/30 border border-border/50 hover:border-primary/30 transition-all duration-300 animate-fade-in"
            style={{ animationDelay: `${0.2 + index * 0.1}s` }}
          >
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 text-primary group-hover:scale-110 transition-transform duration-300">
                <feature.icon className="h-5 w-5" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground text-sm">
                  {feature.title}
                </h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {feature.description}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-12 flex items-center gap-6 text-sm text-muted-foreground animate-fade-in" style={{ animationDelay: "0.6s" }}>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
          <span>All systems operational</span>
        </div>
        <div className="h-4 w-px bg-border" />
        <span>v3.2.1</span>
      </div>
    </div>
  );
};

export default BrandingPanel;
