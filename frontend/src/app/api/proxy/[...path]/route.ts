import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.RSVS_BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.RSVS_API_KEY || '';

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: pathSegments } = await params;
  const backendPath = pathSegments.join('/');
  const body = await request.json();

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }

  try {
    const res = await fetch(`${BACKEND_URL}/${backendPath}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    return NextResponse.json({ error: 'backend_unavailable' }, { status: 502 });
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: pathSegments } = await params;
  const backendPath = pathSegments.join('/');
  const searchParams = request.nextUrl.search;

  const headers: Record<string, string> = {};
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }

  try {
    const res = await fetch(`${BACKEND_URL}/${backendPath}${searchParams}`, {
      method: 'GET',
      headers,
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    return NextResponse.json({ error: 'backend_unavailable' }, { status: 502 });
  }
}
