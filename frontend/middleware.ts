import { createMiddlewareClient } from '@supabase/auth-helpers-nextjs';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function middleware(req: NextRequest) {
  const res = NextResponse.next();
  const supabase = createMiddlewareClient({ req, res });

  const {
    data: { session },
  } = await supabase.auth.getSession();

  // Auth routes that don't require authentication
  const publicAuthRoutes = ['/auth/login', '/auth/signup', '/auth/forgot-password'];
  const isAuthRoute = publicAuthRoutes.includes(req.nextUrl.pathname);

  // Protected routes
  if (req.nextUrl.pathname.startsWith('/sbs-pro') && !session) {
    return NextResponse.redirect(new URL('/auth/login', req.url));
  }

  // Redirect logged-in users away from auth pages
  if (isAuthRoute && session) {
    return NextResponse.redirect(new URL('/sbs-pro/dashboard', req.url));
  }

  return res;
}

export const config = {
  matcher: ['/sbs-pro/:path*', '/auth/:path*'],
}; 