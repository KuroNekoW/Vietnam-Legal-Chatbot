interface SealMarkProps {
  className?: string;
}

export default function SealMark({ className }: SealMarkProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="24" cy="24" r="21" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="24" cy="24" r="15.5" stroke="currentColor" strokeWidth="1" />
      <path
        d="M24 14.5l2.47 5.24 5.73.64-4.27 3.93 1.15 5.69L24 27.1l-5.08 2.9 1.15-5.69-4.27-3.93 5.73-.64L24 14.5z"
        fill="currentColor"
      />
    </svg>
  );
}
