import { supabase } from '@/lib/supabase';

export async function POST(request: Request) {
  try {
    const { email, fullName, profession } = await request.json();

    if (!email || !fullName || !profession) {
      return new Response(JSON.stringify({ 
        error: 'Email, full name, and profession are required' 
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const { data, error } = await supabase
      .from('waitlist')
      .insert([{ 
        email, 
        full_name: fullName,
        profession 
      }]);

    if (error) {
      if (error.code === '23505') {
        return new Response(JSON.stringify({ 
          error: 'Email already registered' 
        }), {
          status: 409,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw error;
    }

    return new Response(JSON.stringify({ 
      success: true, 
      message: 'Successfully joined waitlist',
      data 
    }), {
      status: 201,
      headers: { 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Waitlist error:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error' 
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
} 