import * as React from "react";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

export interface GradientInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  icon?: LucideIcon;
  rightElement?: React.ReactNode;
  error?: boolean;
}

const GradientInput = React.forwardRef<HTMLInputElement, GradientInputProps>(
  ({ className, type, icon: Icon, rightElement, error, ...props }, ref) => {
    return (
      <div
        className={cn(
          "group relative flex items-center rounded-xl bg-secondary/50 border border-border transition-all duration-300",
          "focus-within:border-primary/50 input-glow",
          error && "border-destructive focus-within:border-destructive",
          className
        )}
      >
        {Icon && (
          <div className="pl-4 text-muted-foreground group-focus-within:text-primary transition-colors duration-300">
            <Icon className="h-5 w-5" />
          </div>
        )}
        <input
          type={type}
          className={cn(
            "flex h-12 w-full bg-transparent px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground",
            "focus:outline-none disabled:cursor-not-allowed disabled:opacity-50",
            Icon && "pl-3"
          )}
          ref={ref}
          {...props}
        />
        {rightElement && (
          <div className="pr-3 text-muted-foreground">{rightElement}</div>
        )}
      </div>
    );
  }
);

GradientInput.displayName = "GradientInput";

export { GradientInput };
