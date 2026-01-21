//! Procedural macros for distributed tracing with OpenTelemetry context propagation.
//!
//! Provides the `#[traced]` attribute macro for instrumenting functions with
//! automatic span creation and parent context linking.
//!
//! # Example
//!
//! ```ignore
//! use tracing_macros::traced;
//!
//! #[traced("process_frame", parent = trace_ctx)]
//! fn process_frame(
//!     frame: &Frame,
//!     trace_ctx: Option<&TraceContextBytes>,
//! ) -> Result<()> {
//!     // Function body is automatically wrapped in a span
//!     // linked to the parent trace context
//! }
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Expr, FnArg, Ident, ItemFn, LitStr, Token,
};

/// Arguments for the `#[traced]` attribute.
///
/// Supports:
/// - `#[traced("span_name")]`
/// - `#[traced("span_name", parent = ctx_arg)]`
/// - `#[traced("span_name", parent = ctx_arg, fields(key = value, ...))]`
struct TracedArgs {
    span_name: LitStr,
    parent: Option<Ident>,
    fields: Vec<FieldArg>,
}

struct FieldArg {
    name: Ident,
    value: Option<Expr>,
}

impl Parse for TracedArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let span_name: LitStr = input.parse()?;

        let mut parent = None;
        let mut fields = Vec::new();

        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;

            if input.is_empty() {
                break;
            }

            let ident: Ident = input.parse()?;

            if ident == "parent" {
                input.parse::<Token![=]>()?;
                parent = Some(input.parse()?);
            } else if ident == "fields" {
                let content;
                syn::parenthesized!(content in input);
                let field_list: Punctuated<FieldArg, Token![,]> =
                    content.parse_terminated(FieldArg::parse, Token![,])?;
                fields = field_list.into_iter().collect();
            }
        }

        Ok(TracedArgs {
            span_name,
            parent,
            fields,
        })
    }
}

impl Parse for FieldArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;

        let value = if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            Some(input.parse()?)
        } else {
            None
        };

        Ok(FieldArg { name, value })
    }
}

/// Instruments a function with distributed tracing.
///
/// Creates a span that wraps the entire function body, with optional parent
/// context linking for distributed traces across process boundaries.
///
/// # Arguments
///
/// - `span_name` (required): The name of the span (string literal)
/// - `parent = <ident>` (optional): Name of a function parameter containing
///   `Option<&TraceContextBytes>` to link as parent span
/// - `fields(...)` (optional): Additional fields to record on the span
///
/// # Examples
///
/// Basic usage:
/// ```ignore
/// #[traced("my_operation")]
/// fn do_work() -> Result<()> {
///     // ...
/// }
/// ```
///
/// With parent context propagation:
/// ```ignore
/// #[traced("inference", parent = trace_ctx)]
/// fn run_inference(
///     input: &Tensor,
///     trace_ctx: Option<&TraceContextBytes>,
/// ) -> Result<Output> {
///     // Span is linked to parent trace from another process
/// }
/// ```
///
/// With custom fields:
/// ```ignore
/// #[traced("process_frame", parent = ctx, fields(frame_number, camera_id = camera.id))]
/// fn process(
///     frame_number: u64,
///     camera: &Camera,
///     ctx: Option<&TraceContextBytes>,
/// ) -> Result<()> {
///     // Span includes frame_number and camera_id fields
/// }
/// ```
#[proc_macro_attribute]
pub fn traced(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as TracedArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    match generate_traced_fn(args, input_fn) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generate_traced_fn(args: TracedArgs, input_fn: ItemFn) -> syn::Result<TokenStream2> {
    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = input_fn;

    let span_name = &args.span_name;
    let fn_body = &block.stmts;

    // Build field tokens for the span
    let field_tokens = build_field_tokens(&args.fields, &sig.inputs)?;

    // Generate the span creation and parent linking code
    let span_creation = if args.fields.is_empty() {
        quote! {
            let __traced_span = tracing::info_span!(#span_name);
        }
    } else {
        quote! {
            let __traced_span = tracing::info_span!(#span_name, #field_tokens);
        }
    };

    // Generate parent linking if specified
    let parent_linking = if let Some(parent_ident) = &args.parent {
        quote! {
            if let Some(__parent_ctx) = #parent_ident {
                tracing_opentelemetry::OpenTelemetrySpanExt::set_parent(
                    &__traced_span,
                    __parent_ctx.into_context(),
                );
            }
        }
    } else {
        quote! {}
    };

    let output = quote! {
        #(#attrs)*
        #[allow(unused_must_use)]
        #vis #sig {
            #span_creation
            #parent_linking
            let __traced_guard = __traced_span.enter();

            #(#fn_body)*
        }
    };

    Ok(output)
}

/// Build the field tokens for span creation.
fn build_field_tokens(
    fields: &[FieldArg],
    _inputs: &Punctuated<FnArg, Token![,]>,
) -> syn::Result<TokenStream2> {
    if fields.is_empty() {
        return Ok(quote! {});
    }

    let field_exprs: Vec<TokenStream2> = fields
        .iter()
        .map(|f| {
            let name = &f.name;
            if let Some(value) = &f.value {
                quote! { #name = #value }
            } else {
                // Use the field name as both key and value (shorthand)
                quote! { #name = #name }
            }
        })
        .collect();

    Ok(quote! { #(#field_exprs),* })
}
