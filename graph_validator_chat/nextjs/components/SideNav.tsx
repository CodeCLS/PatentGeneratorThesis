import Link from "next/link";
import styles from "./SideNav.module.css";

type NavPage = "upload" | "analyze" | "edit" | "chat" | "graph" | "editor";

const NAV_ITEMS: { key: NavPage; label: string; href: string }[] = [
  { key: "upload", label: "Upload", href: "/upload" },
  { key: "analyze", label: "Analyze", href: "/analyze" },
  { key: "edit", label: "Edit", href: "/edit" },
  { key: "chat", label: "Chat", href: "/chat" },
  { key: "graph", label: "Graph", href: "/graph" },
  { key: "editor", label: "Editor", href: "/editor" },
];

export default function SideNav({ current }: { current: NavPage }) {
  return (
    <nav className={styles.sidebar} aria-label="Primary">
      <div className={styles.navList}>
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.key}
            href={item.href}
            className={`${styles.navLink} ${current === item.key ? styles.navActive : ""}`}
            aria-current={current === item.key ? "page" : undefined}
          >
            {item.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
