#define IN_sfa

#include "sfa_private.h"

//----------------------------------------------------------------------------//
// Top level function to select and call the proper compute_curl_b function.
//----------------------------------------------------------------------------//

void
compute_curl_b( field_array_t * RESTRICT fa )
{
  if ( !fa )
  {
    ERROR( ( "Bad args" ) );
  }

  // Conditionally execute this when more abstractions are available.
  compute_curl_b_pipeline( fa );
}
