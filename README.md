# SitBlinkSip

This repository is the marketing site for **SitBlinkSip** — a free, open-source desktop app for
developers and other computer-heavy workers. It watches your posture and blink rate through your
webcam, runs quietly in the system tray, and reminds you to sit right, blink bright, and sip well.

The site is a static Next.js app with **no backend** — everything (posture/blink detection, alerts,
storage) runs locally inside the desktop app itself, which lives in a separate repository:
[desktop-sitblinksip](https://github.com/ishworrsubedii/desktop-sitblinksip).

## Problem Statement

Many developers and tech workers often forget to take care of their health while immersed in work.
Common issues include:

- **Poor posture** leading to back and neck pain.
- **Eye strain** and dryness due to a lack of blinking.
- **Dehydration** from forgetting to take water breaks.

SitBlinkSip Desktop tackles these issues by providing timely reminders and real-time, on-device
monitoring to support a healthier work routine — with no camera frame ever leaving your machine.

## Features

- **Real-time Posture Monitoring**: Detects improper posture (slouching or leaning) and alerts users
  to sit correctly.
- **Eye Blink Detection**: Monitors eye blinking, sending a notification if no blink is detected for
  over 60 seconds to prevent dry eyes.
- **Water Break Reminders**: Sends periodic notifications to ensure users stay hydrated.
- **Fully Local**: All computer-vision processing happens on-device. Nothing to host, no account, no
  data sent anywhere.

## Desktop App

[`desktop-sitblinksip`](https://github.com/ishworrsubedii/desktop-sitblinksip) is the standalone
native desktop app: it counts blinks quietly without showing the camera feed, reveals the live
preview on demand with **F6**, and blanks the screen briefly if your blink rate drops too low.

| OS | Status |
| --- | --- |
| Linux | ✅ Available now |
| Windows | 🕒 Coming soon |
| macOS | Not currently planned |

See its [README](https://github.com/ishworrsubedii/desktop-sitblinksip#readme) for setup/packaging
instructions.

## This Repository (Marketing Site)

### Prerequisites

- Node 20+

### Run locally

```bash
git clone https://github.com/ishworrsubedii/SitBlinkSip.git
cd SitBlinkSip
npm install
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

### Build for production

```bash
npm run build
npm start
```

### Project structure

```
app/         Next.js App Router pages (home, docs, faq, blog, preview, cookies)
components/  Shared UI and marketing components
lib/         Small utilities (e.g. GitHub release download counts)
public/      Static assets
```

## Screenshots

### Landing page

![Frontend landing page](demo/frontend-ui.png)

## Author Information

- **Email**: [ishworr.subedi@gmail.com](mailto:ishworr.subedi@gmail.com)
- **GitHub**: [ishworrsubedii](https://github.com/ishworrsubedii)
- **LinkedIn**: [linkedin.com/in/ishworrsubedii](https://www.linkedin.com/in/ishworrsubedii/)
- **Twitter**: [@ishworr_](https://x.com/ishworr_)
- **Portfolio**: [ishwor-subedi.com.np](https://ishwor-subedi.com.np/)

For detailed setup instructions, visit our [Open Source Guidelines](./CONTRIBUTING.md).

---
