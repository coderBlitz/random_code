package main

import (
	"fmt"
	"html"
	"log"
	"net/http"
	"time"
)

func test() {
	// Create default handler
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request){
		fmt.Printf("%q request to URL %q\n", r.Method, r.URL.Path)

		// Default 404 response
		w.WriteHeader(404)
		fmt.Fprintf(w, "404 - URL %q not found", html.EscapeString(r.URL.Path))
	})

	http.HandleFunc("/hi/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Printf("%q request to URL %q\n", r.Method, r.URL.Path)

		fmt.Fprintf(w, "Hello %q, with method %q\n", html.EscapeString(r.URL.Path), r.Method)
	})

	log.Fatal(http.ListenAndServe(":8080", nil))
}

/** Custom handler type to act as wrapper for all other handlers
	TODO: Potentially add map and registration function to struct, to register URLs/paths
	TODO: Call appropriate functions for request URL
**/
type default_srv struct {
}
func (p default_srv) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Printf("%q request to URL %q\n", r.Method, r.URL.Path)
}

func main() {
	fmt.Println("HTTP serv test")

	srv := &http.Server{
		Addr: ":8080",
		Handler: default_srv{},
		ReadTimeout: 5 * time.Second,
		WriteTimeout: 5 * time.Second,
		MaxHeaderBytes: 1 << 13, // Should be 8KiB (1KiB == 2^10)
	}

	log.Fatal(srv.ListenAndServe())
}
