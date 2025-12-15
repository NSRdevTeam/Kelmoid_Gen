import { Shield, Lock, CheckCircle } from "lucide-react";

const TrustBadges = () => {
  const badges = [
    { icon: Shield, text: "SOC 2 Compliant" },
    { icon: Lock, text: "Enterprise Security" },
    { icon: CheckCircle, text: "99.9% Uptime" },
  ];

  return (
    <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-muted-foreground">
      {badges.map((badge, index) => (
        <div key={index} className="flex items-center gap-1.5">
          <badge.icon className="h-3.5 w-3.5" />
          <span>{badge.text}</span>
        </div>
      ))}
    </div>
  );
};

export default TrustBadges;
