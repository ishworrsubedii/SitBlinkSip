const DESKTOP_REPO = "ishworrsubedii/desktop-sitblinksip";

export interface DownloadCounts {
  macos: number;
  windows: number;
  linux: number;
  total: number;
}

interface GitHubAsset {
  name: string;
  download_count: number;
}

interface GitHubRelease {
  assets: GitHubAsset[];
}

/**
 * Lifetime installer download count, summed across every release the
 * desktop repo has ever published.
 *
 * Deliberately grouped by file extension rather than a specific asset name
 * or version - a new tag adds a new release with new asset IDs, so a count
 * tied to one release/asset would reset to zero on every version bump.
 * Extension matching keeps working automatically forever, even if the
 * installer filenames themselves change later.
 *
 * Returns null on any failure (network hiccup, GitHub rate limit, repo
 * renamed) - a download counter is decoration, never worth breaking the
 * page over, so callers should just omit the number when this is null.
 */
export async function getDesktopDownloadCounts(): Promise<DownloadCounts | null> {
  const counts: DownloadCounts = { macos: 0, windows: 0, linux: 0, total: 0 };

  try {
    // GitHub caps a page at 100; loop until a short page signals the end.
    // Capped at 10 pages (1000 releases) as a sane upper bound - this repo
    // will not realistically outgrow that.
    for (let page = 1; page <= 10; page++) {
      const res = await fetch(
        `https://api.github.com/repos/${DESKTOP_REPO}/releases?per_page=100&page=${page}`,
        {
          headers: { Accept: "application/vnd.github+json" },
          // Revalidate hourly: fresh enough for a "downloads" counter
          // without hammering GitHub's unauthenticated 60 req/hour limit.
          next: { revalidate: 3600 },
        }
      );
      if (!res.ok) break;

      const releases: GitHubRelease[] = await res.json();
      if (releases.length === 0) break;

      for (const release of releases) {
        for (const asset of release.assets ?? []) {
          const name = asset.name.toLowerCase();
          if (name.endsWith(".deb")) counts.linux += asset.download_count;
          else if (name.endsWith(".exe")) counts.windows += asset.download_count;
          else if (name.endsWith(".dmg")) counts.macos += asset.download_count;
        }
      }

      if (releases.length < 100) break;
    }
  } catch {
    return null;
  }

  counts.total = counts.macos + counts.windows + counts.linux;
  return counts;
}

/** "1234" -> "1,234", "12500" -> "12.5K" - keeps a big lifetime total glanceable. */
export function formatDownloadCount(n: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: n >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: 1,
  }).format(n);
}
