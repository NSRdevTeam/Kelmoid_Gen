import { cn } from "@/lib/utils";

interface KelmoidLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

const KelmoidLogo = ({ className, size = "md" }: KelmoidLogoProps) => {
  const sizeClasses = {
    sm: "h-8",
    md: "h-10",
    lg: "h-12",
  };

  return (
    <div className={cn("flex items-center gap-3", className)}>
      {/* Logo Mark */}
      <div className={cn("relative", sizeClasses[size])}>
        <div className="absolute inset-0 bg-gradient-to-br from-primary to-accent rounded-xl blur-lg opacity-50" />
        <div className="relative h-full aspect-square bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            className="w-2/3 h-2/3 text-primary-foreground"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
      </div>
      
      {/* Logo Text */}
      <div className="flex flex-col">
        <span className={cn(
          "font-bold tracking-tight leading-none",
          size === "sm" && "text-lg",
          size === "md" && "text-xl",
          size === "lg" && "text-2xl"
        )}>
          <span className="text-foreground">Kelmoid</span>
          <span className="gradient-text ml-1">Genesis</span>
        </span>
        <span className={cn(
          "text-muted-foreground font-medium tracking-wider uppercase",
          size === "sm" && "text-[9px]",
          size === "md" && "text-[10px]",
          size === "lg" && "text-xs"
        )}>
          LLM Platform
        </span>
      </div>
    </div>
  );
};

export default KelmoidLogo;
