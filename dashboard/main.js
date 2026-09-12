/**
 * INTELLIGENCE DESIGNED TO EVOLVE — MAIN SCRIPT
 * Handles mobile drawer interactions and animated count-up statistics.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ============================================================
  // 1. MOBILE DRAWER CONTROLLER
  // ============================================================
  const burgerBtn = document.getElementById("burger-btn");
  const mobileMenu = document.getElementById("mobile-menu");
  const mobileOverlay = document.getElementById("mobile-overlay");
  const mobileNavLinks = document.querySelectorAll(".mobile-nav-link, .mobile-sign-in-btn");
  const desktopNavLinks = document.querySelectorAll(".nav-link");

  let isMenuOpen = false;

  function openMobileMenu() {
    isMenuOpen = true;
    burgerBtn.classList.add("open");
    burgerBtn.setAttribute("aria-expanded", "true");

    mobileOverlay.removeAttribute("hidden");
    mobileMenu.removeAttribute("hidden");
    mobileMenu.setAttribute("aria-hidden", "false");

    // Force reflow for smooth animation
    void mobileOverlay.offsetWidth;
    void mobileMenu.offsetWidth;

    mobileOverlay.classList.add("active");
    mobileMenu.classList.add("active");
    document.body.classList.add("menu-open");
  }

  function closeMobileMenu() {
    isMenuOpen = false;
    burgerBtn.classList.remove("open");
    burgerBtn.setAttribute("aria-expanded", "false");

    mobileOverlay.classList.remove("active");
    mobileMenu.classList.remove("active");
    mobileMenu.setAttribute("aria-hidden", "true");
    document.body.classList.remove("menu-open");

    setTimeout(() => {
      if (!isMenuOpen) {
        mobileOverlay.setAttribute("hidden", "");
        mobileMenu.setAttribute("hidden", "");
      }
    }, 380);
  }

  function toggleMobileMenu() {
    if (isMenuOpen) {
      closeMobileMenu();
    } else {
      openMobileMenu();
    }
  }

  if (burgerBtn && mobileMenu && mobileOverlay) {
    burgerBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleMobileMenu();
    });

    mobileOverlay.addEventListener("click", () => {
      closeMobileMenu();
    });

    // Close on Escape key
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isMenuOpen) {
        closeMobileMenu();
      }
    });

    // Close on resize > 720px
    window.addEventListener("resize", () => {
      if (window.innerWidth > 720 && isMenuOpen) {
        closeMobileMenu();
      }
    });

    // Close on mobile link click & update active state
    mobileNavLinks.forEach((link) => {
      link.addEventListener("click", () => {
        mobileNavLinks.forEach((l) => l.classList.remove("active"));
        if (link.classList.contains("mobile-nav-link")) {
          link.classList.add("active");
        }
        closeMobileMenu();
      });
    });
  }

  // Desktop Nav link active indicator
  desktopNavLinks.forEach((link) => {
    link.addEventListener("click", () => {
      desktopNavLinks.forEach((l) => l.classList.remove("active"));
      link.classList.add("active");
    });
  });

  // ============================================================
  // 2. STATS COUNT-UP ANIMATION
  // ============================================================
  const statItems = document.querySelectorAll(".stat-item");

  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

  function animateValue(element, target, decimals, suffix, duration, delay) {
    setTimeout(() => {
      const startTime = performance.now();
      const startVal = 0;

      function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = easeOutCubic(progress);

        const currentVal = startVal + (target - startVal) * easedProgress;

        if (decimals > 0) {
          element.textContent = currentVal.toFixed(decimals) + suffix;
        } else {
          element.textContent = Math.round(currentVal) + suffix;
        }

        if (progress < 1) {
          requestAnimationFrame(update);
        } else {
          element.textContent = (decimals > 0 ? target.toFixed(decimals) : target) + suffix;
        }
      }

      requestAnimationFrame(update);
    }, delay);
  }

  let hasAnimated = false;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !hasAnimated) {
          hasAnimated = true;

          statItems.forEach((item, index) => {
            const valEl = item.querySelector(".stat-value");
            const target = parseFloat(item.dataset.target || "0");
            const decimals = parseInt(item.dataset.decimals || "0", 10);
            const suffix = item.dataset.suffix || "";

            const duration = 1500 + index * 80;
            const startOffset = 480 + index * 90;

            animateValue(valEl, target, decimals, suffix, duration, startOffset);
          });

          observer.disconnect();
        }
      });
    },
    { threshold: 0.25 }
  );

  const statsFooter = document.querySelector(".stats-footer");
  if (statsFooter) {
    observer.observe(statsFooter);
  }
});
