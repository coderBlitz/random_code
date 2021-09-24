package main

import (
	"fmt"
	"net"
	"os"
	"os/signal"
	"time"
)

func connChan(c chan<- []byte, con *net.UDPConn, N int) {
	buf := make([]byte, N)

	ret, _, err := con.ReadFromUDP(buf)
	for err == nil {
		fmt.Println("Ret =", ret)
		buf = buf[:ret] // Set slice length to the length of returned data
		if ret > 0 {
			c <- buf
		}
		buf = buf[:cap(buf)] // Set slice back to capacity for read

		ret, _, err = con.ReadFromUDP(buf)
		//fmt.Println("Read data =", ret, "from", addr, err)
	}

	//fmt.Println("Done!", err)
}

func main() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

	lnet := "127.0.0.1"
	lport := 2000
	laddr := net.UDPAddr {net.ParseIP(lnet), lport, ""}

	fmt.Println("Listening on", lnet, ":", lport)
	con, err := net.ListenUDP("udp", &laddr)
	if err != nil {
		println("Listen error:", err, err.Error())
	}
	defer con.Close()

	N := 128
	can := make(chan []byte, N)
	go connChan(can, con, N)

	//var ret int
	exit := false
	for !exit {
		select {
			case <- c:
				fmt.Println("Caught signal!")
				exit = true
			case buf := <- can:
				fmt.Println(buf)
			default:
				time.Sleep(20 * time.Millisecond)
		}
	}
}
