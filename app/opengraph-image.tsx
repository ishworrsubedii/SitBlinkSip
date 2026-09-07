import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const alt = 'SitBlinkSip — Sit Right, Blink Bright, Sip Well';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          padding: '80px',
          background: 'linear-gradient(135deg, #eff6ff 0%, #ffffff 60%)',
          fontFamily: 'sans-serif',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            marginBottom: 36,
          }}
        >
          <div
            style={{
              display: 'flex',
              width: 56,
              height: 56,
              borderRadius: 16,
              background: '#2563eb',
            }}
          />
          <div style={{ display: 'flex', fontSize: 40, fontWeight: 700, color: '#0f172a' }}>
            SitBlinkSip
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: 60,
            fontWeight: 800,
            color: '#0f172a',
            lineHeight: 1.15,
            maxWidth: 920,
          }}
        >
          Sit Right. Blink Bright. Sip Well.
        </div>
        <div
          style={{
            display: 'flex',
            marginTop: 28,
            fontSize: 30,
            color: '#475569',
            maxWidth: 860,
          }}
        >
          A free, open-source desktop wellness companion for developers — posture, blink, and hydration reminders that run fully offline.
        </div>
      </div>
    ),
    { ...size }
  );
}
